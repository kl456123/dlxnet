#ifndef DLXNET_CORE_COMMON_RUNTIME_GPU_DEVICE_CONTEXT_H_
#define DLXNET_CORE_COMMON_RUNTIME_GPU_DEVICE_CONTEXT_H_
#include "dlxnet/core/framework/device_base.h"
#include "dlxnet/core/common_runtime/device.h"
#include "dlxnet/core/lib/gtl/inlined_vector.h"

namespace stream_executor {
    class Stream;
}  // namespace stream_executor

namespace dlxnet{
    class GPUDeviceContext : public DeviceContext {
        public:
            GPUDeviceContext(int stream_id)
                :stream_id_(stream_id){}
            ~GPUDeviceContext() override {}

            se::Stream* stream() const override { return stream_; }

            se::Stream* host_to_device_stream() const { return host_to_device_stream_; }
            se::Stream* device_to_host_stream() const { return device_to_host_stream_; }
            se::Stream* device_to_device_stream(int index) const {
                return device_to_device_stream_[index % device_to_device_stream_.size()];
            }

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

            // The default primary stream to use for this context.
            // All the memory belongs to this stream.
            se::Stream* stream_;

            // The stream to use for copying data from host into GPU.
            se::Stream* host_to_device_stream_;
            // The stream to use for copying data from GPU to host.
            se::Stream* device_to_host_stream_;
            // Streams to use for copying data between GPUs.
            gtl::InlinedVector<se::Stream*, 4> device_to_device_stream_;
    };
}//namespace

#endif
