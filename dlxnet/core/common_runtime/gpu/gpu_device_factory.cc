#include "dlxnet/core/common_runtime/gpu/gpu_device.h"


namespace dlxnet{
    class GPUDevice : public BaseGPUDevice {
        public:
            GPUDevice(const SessionOptions& options, const string& name,
                    Bytes memory_limit, const DeviceLocality& locality,
                    TfGpuId tf_gpu_id, const string& physical_device_desc,
                    Allocator* gpu_allocator, Allocator* cpu_allocator)
                : BaseGPUDevice(options, name, memory_limit, locality, tf_gpu_id,
                        physical_device_desc, gpu_allocator, cpu_allocator,
                        false /* sync every op */) {
                    if (options.config.has_gpu_options()) {
                        force_gpu_compatible_ =
                            options.config.gpu_options().force_gpu_compatible();
                    }
                }

            Allocator* GetAllocator(AllocatorAttributes attr) override {
                CHECK(cpu_allocator_) << "bad place 1";
                if (attr.on_host()) {
                    // if (attr.gpu_compatible() || force_gpu_compatible_) {
                    // GPUProcessState* ps = GPUProcessState::singleton();
                    // return ps->GetGpuHostAllocator(0);
                    // } else {
                    return cpu_allocator_;
                    // }
                } else {
                    return gpu_allocator_;
                }
            }
        private:
            bool force_gpu_compatible_ = false;
    };

    class GPUDeviceFactory : public BaseGPUDeviceFactory {
        private:
            std::unique_ptr<BaseGPUDevice> CreateGPUDevice(
                    const SessionOptions& options, const string& name, Bytes memory_limit,
                    const DeviceLocality& locality, TfGpuId tf_gpu_id,
                    const string& physical_device_desc, Allocator* gpu_allocator,
                    Allocator* cpu_allocator) override {
                return absl::make_unique<GPUDevice>(options, name, memory_limit, locality,
                        tf_gpu_id, physical_device_desc,
                        gpu_allocator, cpu_allocator);
            }
    };
    REGISTER_LOCAL_DEVICE_FACTORY("GPU", GPUDeviceFactory, 210);
}//namespace dlxnet
