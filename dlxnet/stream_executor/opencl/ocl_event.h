#ifndef DLXNET_STREAM_EXECUTOR_OCL_OCL_EVENT_H_
#define DLXNET_STREAM_EXECUTOR_OCL_OCL_EVENT_H_
#include "dlxnet/stream_executor/lib/status.h"
#include "dlxnet/stream_executor/stream_executor_internal.h"
#include "dlxnet/stream_executor/opencl/ocl_driver.h"
#include "dlxnet/stream_executor/opencl/ocl_stream.h"

namespace stream_executor{
    class OCLExecutor;

    // GpuEvent wraps a GpuEventHandle in the platform-independent EventInterface
    // interface.
    class GpuEvent : public internal::EventInterface {
        public:
            explicit GpuEvent(OCLExecutor* parent);

            ~GpuEvent() override;

            // Populates the CUDA-platform-specific elements of this object.
            port::Status Init();

            // Deallocates any platform-specific elements of this object. This is broken
            // out (not part of the destructor) to allow for error reporting.
            port::Status Destroy();

            // Inserts the event at the current position into the specified stream.
            port::Status Record(GpuStream* stream);

            // Polls the CUDA platform for the event's current status.
            Event::Status PollForStatus();

            // The underlying CUDA event element.
            GpuEventHandle gpu_event();

        private:
            // The Executor used to which this object and GpuEventHandle are bound.
            OCLExecutor* parent_;

            // The underlying CUDA event element.
            GpuEventHandle gpu_event_;
    };
} // namespace stream_executor



#endif
