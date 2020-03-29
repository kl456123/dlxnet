#ifndef DLXNET_CORE_COMMON_RUNTIME_EXECUTOR_H_
#define DLXNET_CORE_COMMON_RUNTIME_EXECUTOR_H_
#include <functional>
#include "dlxnet/core/framework/function.h"
#include "dlxnet/core/framework/session_state.h"
#include "dlxnet/core/framework/op_kernel.h"
#include "dlxnet/core/framework/node_def.pb.h"
#include "dlxnet/core/lib/core/threadpool_interface.h"
#include "dlxnet/core/lib/core/notification.h"
#include "dlxnet/core/lib/errors.h"
#include "dlxnet/core/common_runtime/device.h"
#include "dlxnet/core/graph/graph.h"

namespace dlxnet{
    class Executor{
        public:
            virtual ~Executor() {}
            struct Args {
                int64 step_id = 0;
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
        std::function<Status(const NodeDef&, OpKernel**)> create_kernel;
        std::function<void(OpKernel*)> delete_kernel;
    };
    ::dlxnet::Status NewLocalExecutor(const LocalExecutorParams& params,
            const Graph& graph, Executor** executor);

    // A few helpers to facilitate create/delete kernels.

    // Creates a kernel based on "ndef" on device "device". The kernel can
    // access the functions in the "flib". The caller takes ownership of
    // returned "*kernel".
    Status CreateNonCachedKernel(Device* device, const NodeDef& ndef,
            int graph_def_version, OpKernel** kernel);

    // Deletes "kernel" returned by CreateKernel.
    void DeleteNonCachedKernel(OpKernel* kernel);
}// namespace dlxnet


#endif
