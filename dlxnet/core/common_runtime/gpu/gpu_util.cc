#include "dlxnet/core/common_runtime/gpu/gpu_util.h"
#include "dlxnet/core/common_runtime/gpu_device_context.h"
#include "dlxnet/core/common_runtime/dma_helper.h"
#include "dlxnet/core/platform/stream_executor.h"

namespace dlxnet{
    using se::DeviceMemoryBase;
    using se::Stream;

    Status PrepareCopy(Device* device, const DeviceContext* ctx, const Tensor& src,
            const Tensor* dst,
            const DeviceBase::DeviceInfo** dev_info,
            se::Stream** stream) {
        if (device == nullptr) {
            return errors::Internal("Unexpected null device.");
        }
        auto di = device->device_info();
        if (di == nullptr) {
            return errors::Internal("Unexpected null device info.");
        }
        *dev_info = di;
        if (ctx == nullptr) {
            return errors::Internal("Unexpected null device context.");
        }
        auto gs = static_cast<const GPUDeviceContext*>(ctx)->stream();
        if (gs == nullptr) {
            return errors::Internal("No gpu stream is available.");
        }
        *stream = gs;
        if (dst != nullptr) {
            if (src.dtype() != dst->dtype()) {
                return errors::Internal("Can't copy a tensor of ",
                        DataTypeString(src.dtype()), " into a tensor of ",
                        DataTypeString(dst->dtype()));
            }
            if (src.TotalBytes() != dst->TotalBytes()) {
                return errors::Internal("Can't copy ", src.TotalBytes(),
                        " bytes of a tensor into another with ",
                        dst->TotalBytes(), " bytes buffer.");
            }
            if ((src.TotalBytes() > 0) && !src.IsInitialized()) {
                return errors::Internal("Src tensor is not initialized.");
            }
        }

        if (!DMAHelper::CanUseDMA(&src)) {
            return errors::Internal("GPU copy from non-DMA ",
                    DataTypeString(src.dtype()), " tensor");
        }
        return Status::OK();
    }

    void* GetBase(const Tensor* src) {
        return const_cast<void*>(DMAHelper::base(src));
    }

    void* GetBase(Tensor* dst) { return DMAHelper::base(dst); }

    Status GPUUtil::Sync(Device* gpu_device) {
        VLOG(1) << "GPUUtil::Sync";
        auto* dev_info = gpu_device->device_info();
        if (!dev_info) {
            return errors::Internal("Failed to find dest device GPUDeviceInfo");
        }
        return dev_info->stream->BlockHostUntilDone();
    }

    Status GPUUtil::SyncAll(Device* gpu_device) {
        VLOG(1) << "GPUUtil::SyncAll";
        auto* dev_info = gpu_device->device_info();
        if (!dev_info) {
            return errors::Internal("Failed to find dest device GPUDeviceInfo");
        }
        // if (!dev_info->stream->parent()->SynchronizeAllActivity() ||
        // !dev_info->stream->ok()) {
        // return errors::Internal("GPU sync failed");
        // }
        return Status::OK();
    }

    // static
    void GPUUtil::CopyGPUTensorToSameGPU(Device* gpu_device,
            const DeviceContext* device_context,
            const Tensor* src_gpu_tensor,
            Tensor* dst_gpu_tensor,
            StatusCallback done) {
        VLOG(1) << "CopyGPUTensorToSameGPU";
    }

    // static
    void GPUUtil::CopyGPUTensorToCPU(Device* gpu_device,
            const DeviceContext* device_context,
            const Tensor* gpu_tensor, Tensor* cpu_tensor,
            StatusCallback done) {
        VLOG(1) << "CopyGPUTensorToCPU";
        const DeviceBase::DeviceInfo* dev_info = nullptr;
        se::Stream* send_stream = nullptr;
        Status s = PrepareCopy(gpu_device, device_context, *gpu_tensor, cpu_tensor,
                &dev_info, &send_stream);
        if (!s.ok()) {
            done(s);
            return;
        }

        auto send_device_to_host_stream =
            static_cast<const GPUDeviceContext*>(device_context)
            ->device_to_host_stream();
        if (send_device_to_host_stream == nullptr) {
            done(errors::Internal("No send gpu copy-out-stream is available."));
            return;
        }
        // Wait for the sender's main stream to make sure the data are available.
        send_device_to_host_stream->ThenWaitFor(send_stream);

        const int64 total_bytes = gpu_tensor->TotalBytes();
        if (total_bytes > 0) {
            void* src_ptr = GetBase(gpu_tensor);
            DeviceMemoryBase gpu_src_ptr(src_ptr, total_bytes);
            void* dst_ptr = GetBase(cpu_tensor);
            send_device_to_host_stream->ThenMemcpy(dst_ptr, gpu_src_ptr, total_bytes);
        }
        if (!send_device_to_host_stream->ok()) {
            LOG(FATAL) << "GPU->CPU Memcpy failed";
        }
        done(Status::OK());
    }

    /*  static */
    void GPUUtil::CopyCPUTensorToGPU(const Tensor* cpu_tensor,
            const DeviceContext* device_context,
            Device* gpu_device, Tensor* gpu_tensor,
            StatusCallback done, bool sync_dst_compute) {
        VLOG(1) << "CopyCPUTensorToGPU";

        const DeviceBase::DeviceInfo* dev_info = nullptr;
        se::Stream* recv_stream = nullptr;
        Status s = PrepareCopy(gpu_device, device_context, *cpu_tensor, gpu_tensor,
                &dev_info, &recv_stream);
        if (!s.ok()) {
            done(s);
            return;
        }

        auto recv_host_to_device_stream =
            static_cast<const GPUDeviceContext*>(device_context)
            ->host_to_device_stream();
        if (recv_host_to_device_stream == nullptr) {
            done(errors::Internal("No send gpu copy-out-stream is available."));
            return;
        }
        // Wait for the recv-stream to make sure the buffer is truly available.
        if (sync_dst_compute) {
            recv_host_to_device_stream->ThenWaitFor(recv_stream);
        }

        const int64 total_bytes = cpu_tensor->TotalBytes();
        // Note that 0-size tensors have no backing buffer.
        if (total_bytes > 0) {
            void* src_ptr = GetBase(cpu_tensor);
            void* dst_ptr = GetBase(gpu_tensor);
            DeviceMemoryBase gpu_dst_ptr(dst_ptr, total_bytes);
            recv_host_to_device_stream->ThenMemcpy(&gpu_dst_ptr, src_ptr, total_bytes);
        }

        if (!recv_host_to_device_stream->ok()) {
            LOG(FATAL) << "CPU->GPU Memcpy failed";
        }
        done(Status::OK());
    }
}// namespace dlxnet
