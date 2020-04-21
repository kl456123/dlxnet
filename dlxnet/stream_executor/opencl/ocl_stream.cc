#include "dlxnet/stream_executor/opencl/ocl_stream.h"
#include "dlxnet/stream_executor/opencl/ocl_gpu_executor.h"
#include "dlxnet/stream_executor/lib/status.h"
#include "dlxnet/stream_executor/stream.h"

namespace stream_executor {
    bool GpuStream::Init() {
        if (!OCLDriver::CreateStream(parent_->gpu_context(), &gpu_stream_)) {
            return false;
        }
        // return GpuDriver::InitEvent(parent_->gpu_context(), &completed_event_,
        // GpuDriver::EventFlags::kDisableTiming)
        // .ok();
        return true;
    }

    void GpuStream::Destroy() {
        if (completed_event_() != nullptr) {
            port::Status status = port::Status::OK();
            if (!status.ok()) {
                LOG(ERROR) << status.error_message();
            }

        }
    }

    bool GpuStream::IsIdle() const {return true;}

    GpuStream* AsGpuStream(Stream* stream) {
        DCHECK(stream != nullptr);
        return static_cast<GpuStream*>(stream->implementation());
    }

    GpuStreamHandle AsGpuStreamValue(Stream* stream) {
        DCHECK(stream != nullptr);
        return AsGpuStream(stream)->gpu_stream();
    }
}


