#ifndef DLXNET_STREAM_EXECUTOR_STREAM_H_
#define DLXNET_STREAM_EXECUTOR_STREAM_H_
#include <functional>
#include <memory>

#include "absl/synchronization/mutex.h"
#include "dlxnet/core/platform/macros.h"
#include "dlxnet/stream_executor/device_memory.h"
#include "dlxnet/stream_executor/event.h"
#include "dlxnet/stream_executor/kernel.h"
#include "dlxnet/stream_executor/launch_dim.h"
#include "dlxnet/stream_executor/lib/array_slice.h"
#include "dlxnet/stream_executor/lib/status.h"
#include "dlxnet/stream_executor/platform/port.h"
#include "dlxnet/stream_executor/platform/thread_annotations.h"




namespace stream_executor{
    namespace internal {
        class StreamInterface;
    }  // namespace internal

    class DeviceMemoryBase;
    template <typename ElemT>
        class DeviceMemory;

    class Timer;

    class StreamExecutor;

    // Represents a stream of dependent computations on a GPU device.
    //
    // The operations within a stream execute linearly and asynchronously until
    // BlockHostUntilDone() is invoked, which synchronously joins host code with
    // the execution of the stream.
    //
    // If any given operation fails when entraining work for the stream, ok() will
    // indicate that an error has occurred. After initialization, once a stream is
    // !ok(), it will never be ok().
    //
    // Thread-safe post-initialization.
    class Stream {
        public:
            // Instantiate a stream tied to parent as a platform executor. Work
            // entrained onto this stream will be launched/managed on that
            // StreamExecutor's platform.
            explicit Stream(StreamExecutor *parent);

            // Test only. Use an externally-populated value (like a mock) for the
            // platform-specific stream implementation.
            Stream(StreamExecutor *parent, internal::StreamInterface *implementation);

            // Deallocates any stream resources that the parent StreamExecutor has
            // bestowed
            // upon this object.
            ~Stream();

            // Returns whether any errors have occurred while entraining work for this
            // stream.
            bool ok() const { return !InErrorState(); }

            // Initialize the stream. This must be performed before entraining any other
            // operations.
            Stream &Init() LOCKS_EXCLUDED(mu_);

            // Initializes timer t via the StreamExecutor.
            Stream &InitTimer(Timer *t);

            // (Synchronously) block the host code waiting for the operations
            // entrained on the stream (enqueued to this point in program
            // execution) to complete.
            //
            // Returns an OK status if the blocking was successful and the stream is ok().
            // Otherwise returns an error describing why the blocking failed.
            port::Status BlockHostUntilDone() LOCKS_EXCLUDED(mu_);

            // Entrains onto the stream of operations: a kernel launch with the given
            // (variadic) parameters for the invocation. These arguments can be things
            // like DeviceMemory or primitive types such as int. What arguments you may
            // pass to a given kernel are noted as the template parameters to the
            // TypedKernel type that the machocc compiler generates.
            //
            // Template parameters:
            //  Params...   The type list of formal parameters that the typed kernel
            //              expects, which is matched against Args...
            //  Args...     The deduced type list for passed actual arguments
            //
            // Implementation: A compile-time compatibility check is performed that has
            // some leniency versus an exact parameter pack match -- for example,
            // `const DeviceMemory<T>` is considered "pack compatible" with a
            // `const DeviceMemory<T>&` formal parameter; in part, because we don't have
            // perfect forwarding support without rvalue references. It also attempts to
            // spit out helpful static_assert error traces with information as to the
            // argument number and types that were mismatched.
            template <typename... Params, typename... Args>
                Stream &ThenLaunch(ThreadDim thread_dims, BlockDim block_dims,
                        const TypedKernel<Params...> &kernel, Args... args);

            // Record a "start" event for the interval timer at this point in the
            // stream's execution (relative to the previously and subsequently enqueued
            // items in the stream's execution). Streams may be started/stopped multiple
            // times.
            Stream &ThenStartTimer(Timer *t);

            // Record a "stop" event for the interval timer at this point in the
            // stream's execution. See also Stream::ThenStartTimer.
            Stream &ThenStopTimer(Timer *t);

            // Returns the (opaque) platform-specific backing object. Ownership is not
            // transferred to the caller.
            internal::StreamInterface *implementation() { return implementation_.get(); }
            // Returns the StreamExecutor (parent object) associated with this stream.
            StreamExecutor *parent() const {
                CHECK(parent_ != nullptr);
                return parent_;
            }
        private:
            // Sets the error state if operation_retcode is false.
            // This is a useful shorthand for many stream routines.
            void CheckError(bool operation_retcode) LOCKS_EXCLUDED(mu_) {
                if (operation_retcode) {
                    return;
                }
                absl::MutexLock lock(&mu_);
                ok_ = false;
            }

            bool InErrorState() const LOCKS_EXCLUDED(mu_) {
                absl::ReaderMutexLock lock(&mu_);
                return !ok_;
            }
            // Checks the status and logs the error message, if any.
            void CheckStatus(port::Status status) LOCKS_EXCLUDED(mu_);

            void SetError() { CheckError(false /* = operation_retcode */); }

            // The StreamExecutor that supports the operation of this stream.
            StreamExecutor *parent_;

            // The platform-dependent implementation that the StreamExecutor interface
            // delegates to.
            std::unique_ptr<internal::StreamInterface> implementation_;

            // mutex that guards the allocation / error state flags.
            // Mutable so that it can be obtained via const reader lock.
            mutable absl::Mutex mu_;

            // Whether Init() was successfully called to allocate this stream on the
            // underlying platform. It simply flips from 0 to 1 with a sanity check.
            // See StreamExecutor::AllocateStream.
            bool allocated_ GUARDED_BY(mu_);

            // Whether all operations have entrained successfully to the current program
            // point.
            bool ok_ GUARDED_BY(mu_);
            SE_DISALLOW_COPY_AND_ASSIGN(Stream);
    };


}//namespace stream_executor


#endif
