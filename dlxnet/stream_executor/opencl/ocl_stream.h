#ifndef DLXNET_STREAM_EXECUTOR_OCL_OCL_STREAM_H_
#define DLXNET_STREAM_EXECUTOR_OCL_OCL_STREAM_H_
#include "dlxnet/stream_executor/opencl/ocl_driver.h"
#include "dlxnet/stream_executor/platform/thread_annotations.h"
#include "dlxnet/stream_executor/stream_executor_internal.h"

namespace stream_executor{
    class OCLExecutor;

    // Wraps a GpuStreamHandle in order to satisfy the platform-independent
    // StreamInterface.
    //
    // Thread-safe post-initialization.
    class GpuStream : public internal::StreamInterface {
        public:
            explicit GpuStream(OCLExecutor* parent)
                : parent_(parent), gpu_stream_(), completed_event_() {}

            // Note: teardown is handled by a parent's call to DeallocateStream.
            ~GpuStream() override {}

            // Explicitly initialize the CUDA resources associated with this stream, used
            // by StreamExecutor::AllocateStream().
            bool Init();

            // Explicitly destroy the CUDA resources associated with this stream, used by
            // StreamExecutor::DeallocateStream().
            void Destroy();

            // Returns true if no work is pending or executing on the stream.
            bool IsIdle() const;

            // Retrieves an event which indicates that all work enqueued into the stream
            // has completed. Ownership of the event is not transferred to the caller, the
            // event is owned by this stream.
            GpuEventHandle* completed_event() { return &completed_event_; }

            // Returns the GpuStreamHandle value for passing to the CUDA API.
            //
            // Precond: this GpuStream has been allocated (otherwise passing a nullptr
            // into the NVIDIA library causes difficult-to-understand faults).
            GpuStreamHandle gpu_stream() const {
                DCHECK(gpu_stream_() != nullptr);
                return gpu_stream_;
            }

            OCLExecutor* parent() const { return parent_; }
        private:
            OCLExecutor* parent_;         // Executor that spawned this stream.
            GpuStreamHandle gpu_stream_;  // Wrapped CUDA stream handle.

            // Event that indicates this stream has completed.
            GpuEventHandle completed_event_;
    };

    // Helper functions to simplify extremely common flows.
    // Converts a Stream to the underlying GpuStream implementation.
    GpuStream* AsGpuStream(Stream* stream);

    // Extracts a GpuStreamHandle from a GpuStream-backed Stream object.
    GpuStreamHandle AsGpuStreamValue(Stream* stream);

}//namespace stream_executor

#endif
