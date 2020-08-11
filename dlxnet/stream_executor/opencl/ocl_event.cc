#include "dlxnet/stream_executor/opencl/ocl_event.h"
#include "dlxnet/stream_executor/opencl/ocl_gpu_executor.h"


namespace stream_executor{
    GpuEvent::GpuEvent(OCLExecutor* parent)
        : parent_(parent), gpu_event_(nullptr) {}

    GpuEvent::~GpuEvent() {}

    port::Status GpuEvent::Init() {
        return OCLDriver::InitEvent(parent_->gpu_context(), &gpu_event_);
    }

    port::Status GpuEvent::Destroy() {
        return OCLDriver::DestroyEvent(parent_->gpu_context(), &gpu_event_);
    }

    port::Status GpuEvent::Record(GpuStream* stream) {
        return OCLDriver::RecordEvent(parent_->gpu_context(), gpu_event_,
                stream->gpu_stream());
    }

    GpuEventHandle GpuEvent::gpu_event() { return gpu_event_; }

    Event::Status GpuEvent::PollForStatus() {
        // port::StatusOr<GpuStatus> status =
            // GpuDriver::QueryEvent(parent_->gpu_context(), gpu_event_);
        // if (!status.ok()) {
            // LOG(ERROR) << "Error polling for event status: "
                // << status.status().error_message();
            // return Event::Status::kError;
        // }

        // switch (status.ValueOrDie()) {
            // case CUDA_SUCCESS:
                // return Event::Status::kComplete;
            // case CUDA_ERROR_NOT_READY:
                // return Event::Status::kPending;
            // default:
                // LOG(INFO) << "Error condition returned for event status: "
                    // << status.ValueOrDie();
                // return Event::Status::kError;
        // }
    }
} // namespace stream_executor
