#ifndef DLXNET_CORE_COMMON_RUNTIME_GPU_GPU_DEVICE_H_
#define DLXNET_CORE_COMMON_RUNTIME_GPU_GPU_DEVICE_H_
#include <unordered_map>

#include "dlxnet/core/common_runtime/local_device.h"
#include "dlxnet/core/common_runtime/device_factory.h"
#include "dlxnet/core/common_runtime/gpu/gpu_id.h"
#include "dlxnet/core/common_runtime/gpu_device_context.h"
#include "dlxnet/core/common_runtime/gpu/gpu_id_manager.h"
#include "dlxnet/core/platform/stream_executor.h"

namespace dlxnet{
    class BaseGPUDevice: public LocalDevice{
        public:
            BaseGPUDevice(const SessionOptions& options, const string& name,
                    Bytes memory_limit, const DeviceLocality& locality,
                    TfGpuId tf_gpu_id, const string& physical_device_desc,
                    Allocator* gpu_allocator, Allocator* cpu_allocator,
                    bool sync_every_op);

            ~BaseGPUDevice() override;

            // Initialize the device and return the status of initialization.
            Status Init(const SessionOptions& options);

            void Compute(OpKernel* op_kernel, OpKernelContext* context) override;

            Status Sync() override;

            void ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                    AsyncOpKernel::DoneCallback done) override;

            Status MakeTensorFromProto(const TensorProto& tensor_proto,
                    const AllocatorAttributes alloc_attrs,
                    Tensor* tensor) override;

            void CopyTensorInSameDevice(const Tensor* input_tensor, Tensor* output_tensor,
                    const DeviceContext* device_context,
                    StatusCallback done) override;

            // Returns the platform GPU id of this device within the native driver system;
            // e.g., for CUDA and ROCm this is the ordinal of the GPU within the system.
            int gpu_id() const {
                PlatformGpuId platform_gpu_id;
                TF_CHECK_OK(GpuIdManager::TfToPlatformGpuId(tf_gpu_id_, &platform_gpu_id));
                return platform_gpu_id;
            }
        protected:
            Allocator* gpu_allocator_;  // not owned
            Allocator* cpu_allocator_;  // not owned

            se::StreamExecutor* executor_;  // not owned

        private:
            struct StreamGroup {
                se::Stream* compute = nullptr;
                se::Stream* host_to_device = nullptr;
                se::Stream* device_to_host = nullptr;
                gtl::InlinedVector<se::Stream*, 4> device_to_device;
            };
            class StreamGroupFactory;

            StreamGroup* stream_;
            GPUDeviceContext* device_context_;
            DeviceInfo* device_info_ = nullptr;
            std::unique_ptr<thread::ThreadPool> thread_pool_;
            TfGpuId tf_gpu_id_;
            const bool sync_every_op_ = false;
            bool timestamped_allocator_ = false;

            // This method returns an initialization status, in addition to
            // calling the "done" StatusCallback, if there is a failure to
            // allocate memory or if the tensor "from" is not DMA-copyable.
            // If there is no error prior to enqueueing the copy, an OK status
            // is returned.
            Status MaybeCopyTensorToGPU(const AllocatorAttributes& alloc_attrs,
                    const Tensor& from, Tensor* to,
                    StatusCallback done);
            string ComputeOpKernelDebugString(const OpKernel& op_kernel,
                    const int& stream_id);

    };

    class BaseGPUDeviceFactory:public DeviceFactory{
        public:
            Status ListPhysicalDevices(std::vector<string>* devices) override;
            Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                    std::vector<std::unique_ptr<Device>>* devices) override;

        protected:
            typedef std::unordered_map<TfGpuId, DeviceLocality> LocalityMap;
            // Populates *localities with the DeviceLocality descriptor for
            // every TfGpuId.
            virtual Status GetDeviceLocalities(int num_tf_gpus, LocalityMap* localities);
        private:
            // Creates a BaseGPUDevice associated with 'tf_gpu_id', allocates (strictly)
            // 'memory_limit' bytes of GPU memory to it, and adds it to the 'devices'
            // vector.
            Status CreateGPUDevice(const SessionOptions& options,
                    const string& name_prefix, TfGpuId tf_gpu_id,
                    int64 memory_limit, const DeviceLocality& dev_locality,
                    std::vector<std::unique_ptr<Device>>* devices);

            virtual std::unique_ptr<BaseGPUDevice> CreateGPUDevice(
                    const SessionOptions& options, const string& name, Bytes memory_limit,
                    const DeviceLocality& dev_locality, TfGpuId tf_gpu_id,
                    const string& physical_device_desc, Allocator* gpu_allocator,
                    Allocator* cpu_allocator) = 0;

            // Returns into 'ids' the list of valid platform GPU ids, in the order that
            // they should map to TF GPU ids "/device:GPU:0", "/device:GPU:1", etc,
            // based upon 'visible_gpu_order' which was generated by parsing
            // GPUOptions::visible_device_list which is a comma-separated list of CUDA or
            // ROCm GPU ids.
            Status GetValidDeviceIds(const std::vector<PlatformGpuId>& visible_gpu_order,
                    std::vector<PlatformGpuId>* ids);

            // visible_gpu_initialized_[platform_gpu_id] is true if visible GPU
            // platform_gpu_id has been initialized by the process.
            std::unordered_map<int, bool> visible_gpu_initialized_;
    };


}// namespace dlxnet


#endif
