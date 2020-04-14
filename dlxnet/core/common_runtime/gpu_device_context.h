#ifndef DLXNET_CORE_COMMON_RUNTIME_GPU_DEVICE_CONTEXT_H_
#define DLXNET_CORE_COMMON_RUNTIME_GPU_DEVICE_CONTEXT_H_
#include "dlxnet/core/framework/device_base.h"
#include "dlxnet/core/common_runtime/device.h"

namespace dlxnet{
    class GPUDeviceContext : public DeviceContext {
        public:
            GPUDeviceContext(int stream_id)
                :stream_id_(stream_id){}
            ~GPUDeviceContext() override {}

            void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                    Tensor* device_tensor, StatusCallback done,
                    bool sync_dst_compute) const override;

            void CopyDeviceTensorToCPU(const Tensor* device_tensor, StringPiece edge_name,
                    Device* device, Tensor* cpu_tensor,
                    StatusCallback done) override;

            void CopyTensorInSameDevice(const Tensor* input_tensor, Device* device,
                    Tensor* output_tensor,
                    StatusCallback done) const override;

            int stream_id() const { return stream_id_; }
        private:
            int stream_id_;
    };
}//namespace

#endif
