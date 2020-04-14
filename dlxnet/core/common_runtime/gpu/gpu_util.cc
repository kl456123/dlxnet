#include "dlxnet/core/common_runtime/gpu/gpu_util.h"

namespace dlxnet{
    Status GPUUtil::Sync(Device* gpu_device) {
        VLOG(1) << "GPUUtil::Sync";
        // auto* dev_info = gpu_device->tensorflow_gpu_device_info();
        // if (!dev_info) {
        // return errors::Internal("Failed to find dest device GPUDeviceInfo");
        // }
        // return dev_info->stream->BlockHostUntilDone();
        return Status::OK();
    }

    Status GPUUtil::SyncAll(Device* gpu_device) {
        VLOG(1) << "GPUUtil::SyncAll";
        // auto* dev_info = gpu_device->tensorflow_gpu_device_info();
        // if (!dev_info) {
        // return errors::Internal("Failed to find dest device GPUDeviceInfo");
        // }
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
    }

    /*  static */
    void GPUUtil::CopyCPUTensorToGPU(const Tensor* cpu_tensor,
            const DeviceContext* device_context,
            Device* gpu_device, Tensor* gpu_tensor,
            StatusCallback done, bool sync_dst_compute) {
        VLOG(1) << "CopyCPUTensorToGPU";
    }
}// namespace dlxnet
