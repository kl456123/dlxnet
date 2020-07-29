#ifndef DLXNET_CORE_COMMON_RUNTIME_EXECUTOR_H_
#define DLXNET_CORE_COMMON_RUNTIME_EXECUTOR_H_
#include <functional>
#include "dlxnet/core/framework/function.h"
#include "dlxnet/core/framework/session_state.h"
#include "dlxnet/core/framework/op_kernel.h"
#include "dlxnet/core/framework/node_def.pb.h"
#include "dlxnet/core/lib/core/threadpool_interface.h"
#include "dlxnet/core/lib/core/notification.h"
#include "dlxnet/core/lib/core/errors.h"
#include "dlxnet/core/common_runtime/device.h"
// #include "dlxnet/core/common_runtime/rendezvous_mgr.h"
#include "dlxnet/core/framework/rendezvous.h"
#include "dlxnet/core/graph/graph.h"

namespace dlxnet{
    class Executor{
        public:
            virtual ~Executor() {}
            typedef std::function<Status(const int64, const DeviceMgr*, Rendezvous** r)>
                RendezvousFactory;
            struct Args {
                int64 step_id = 0;
                RendezvousInterface* rendezvous = nullptr;
                CallFrameInterface* call_frame = nullptr;
                SessionState* session_state = nullptr;
                // Unique session identifier. Can be empty.
                string session_handle;
                thread::ThreadPoolInterface* user_intra_op_threadpool = nullptr;

                // If true, calls Sync() on the device.
                bool sync_on_finish = false;

                typedef std::function<void()> Closure;
                typedef std::function<void(Closure)> Runner;
                Runner runner = nullptr;
            };
            typedef std::function<void(const Status&)> DoneCallback;
            virtual void RunAsync(const Args& args, DoneCallback done) = 0;
            // Synchronous wrapper for RunAsync().
            virtual Status Run(const Args& args) {
                Status ret;
                Notification n;
                RunAsync(args, [&ret, &n](const Status& s) {
                        ret = s;
                        n.Notify();
                        });
                n.WaitForNotification();
                return ret;
            }
    };

    // Creates an Executor that computes the given "graph".
    //
    // If successful, returns the constructed executor in "*executor". Otherwise,
    // returns an error status.
    //
    // "params" provides a set of context for the executor. We expect that
    // different context would provide different implementations.
    struct LocalExecutorParams {
        Device* device;

        // create_kernel returns an instance of op kernel based on NodeDef.
        // delete_kernel is called for every kernel used by the executor
        // when the executor is deleted.
        std::function<Status(Device* device, const NodeDef&, OpKernel**)> create_kernel;
        Executor::RendezvousFactory rendezvous_factory;
        std::function<void(OpKernel*)> delete_kernel;
    };
    ::dlxnet::Status NewLocalExecutor(const LocalExecutorParams& params,
            const Graph& graph, Executor** executor);

    // A class to help run multiple executors in parallel and wait until
    // all of them are complete.
    //
    // ExecutorBarrier deletes itself after the function returned by Get()
    // is called.
    class ExecutorBarrier {
        public:
            typedef std::function<void(const Status&)> StatusCallback;

            // Create an ExecutorBarrier for 'num' different executors.
            //
            // 'r' is the shared Rendezvous object that is used to communicate
            // state.  If any of the executors experiences an error, the
            // rendezvous object will be aborted exactly once.
            //
            // 'done' is called after the last executor completes, and
            // ExecutorBarrier is deleted.
            ExecutorBarrier(size_t num, Rendezvous* r, StatusCallback done)
                : rendez_(r), done_cb_(done), pending_(num) {}

            ~ExecutorBarrier() {}

            // Returns a closure that Executors must call when they are done
            // computing, passing the status of their execution as an argument.
            StatusCallback Get() {
                return std::bind(&ExecutorBarrier::WhenDone, this, std::placeholders::_1);
            }

        private:
            Rendezvous* rendez_ = nullptr;
            StatusCallback done_cb_ = nullptr;
            mutable mutex mu_;
            int pending_ GUARDED_BY(mu_) = 0;

            void WhenDone(const Status& s) {
                Rendezvous* error_rendez = nullptr;
                StatusCallback done = nullptr;
                Status status;

                {
                    mutex_lock l(mu_);

                    // If we are the first error encountered, trigger an abort of the
                    // Rendezvous object by this thread only.
                    if (!s.ok()) {
                        error_rendez = rendez_;
                        error_rendez->Ref();
                    }

                    // If this is the last call to WhenDone, call the final callback
                    // below.
                    if (--pending_ == 0) {
                        CHECK(done_cb_ != nullptr);
                        std::swap(done, done_cb_);
                        status = s;
                    }
                }

                if (error_rendez != nullptr) {
                    error_rendez->StartAbort(
                            errors::Aborted("Stopping remaining executors."));
                    error_rendez->Unref();
                }

                if (done != nullptr) {
                    delete this;
                    if (!status.ok()) {
                        VLOG(1) << "ExecutorBarrier finished with bad status: " << status;
                    }
                    done(status);
                }
            }
    };

    // A few helpers to facilitate create/delete kernels.

    // Creates a kernel based on "ndef" on device "device". The kernel can
    // access the functions in the "flib". The caller takes ownership of
    // returned "*kernel".
    Status CreateNonCachedKernel(Device* device, const NodeDef& ndef, OpKernel** kernel);

    // Deletes "kernel" returned by CreateKernel.
    void DeleteNonCachedKernel(OpKernel* kernel);
}// namespace dlxnet


#endif
