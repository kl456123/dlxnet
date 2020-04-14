#include "dlxnet/core/common_runtime/gpu_device_context.h"
#include "dlxnet/core/common_runtime/gpu/gpu_util.h"

namespace dlxnet{
    void GPUDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
            Device* device,
            Tensor* device_tensor,
            StatusCallback done,
            bool sync_dst_compute) const {
        GPUUtil::CopyCPUTensorToGPU(cpu_tensor, this, device, device_tensor, done,
                sync_dst_compute);
    }

    void GPUDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
            StringPiece tensor_name,
            Device* device, Tensor* cpu_tensor,
            StatusCallback done) {
        GPUUtil::CopyGPUTensorToCPU(device, this, device_tensor, cpu_tensor, done);
    }

    void GPUDeviceContext::CopyTensorInSameDevice(const Tensor* input_tensor,
            Device* device,
            Tensor* output_tensor,
            StatusCallback done) const {
        GPUUtil::CopyGPUTensorToSameGPU(device, this, input_tensor, output_tensor,
                done);
    }


} // namespace dlxnet
