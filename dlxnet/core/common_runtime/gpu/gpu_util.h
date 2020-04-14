#ifndef DLXNET_CORE_COMMON_RUNTIME_GPU_GPU_UTIL_H_
#define DLXNET_CORE_COMMON_RUNTIME_GPU_GPU_UTIL_H_
#include "dlxnet/core/lib/status.h"
#include "dlxnet/core/common_runtime/device.h"

namespace dlxnet{
    class GPUUtil{
        public:
            // Blocks until all operations queued on all streams associated with the
            // corresponding GPU device at the time of call have completed.
            // Returns any error pending on the stream at completion.
            static Status SyncAll(Device* gpu_device);

            // Blocks until all operations queued on the stream associated with
            // "gpu_device" at the time of the call have completed.  Returns any
            // error pending on the stream at completion.
            static Status Sync(Device* gpu_device);

            // Copies the data in 'gpu_tensor' into 'cpu_tensor'.
            // 'gpu_tensor''s backing memory must be on 'gpu_device' and
            // 'cpu_tensor' must be allocated to be of the same size as
            // 'gpu_tensor'. Synchronous: may block.
            static void CopyGPUTensorToCPU(Device* gpu_device,
                    const DeviceContext* device_context,
                    const Tensor* gpu_tensor, Tensor* cpu_tensor,
                    StatusCallback done);

            static void CopyCPUTensorToGPU(const Tensor* cpu_tensor,
                    const DeviceContext* device_context,
                    Device* gpu_device, Tensor* gpu_tensor,
                    StatusCallback done, bool sync_dst_compute);

            // Deep-copying of GPU tensor on the same device.
            // 'src_gpu_tensor''s and 'dst_gpu_tensor''s backing memory must be on
            // 'gpu_device' and 'dst_cpu_tensor' must be allocated to be of the same
            // size as 'src_gpu_tensor'.
            static void CopyGPUTensorToSameGPU(Device* gpu_device,
                    const DeviceContext* device_context,
                    const Tensor* src_gpu_tensor,
                    Tensor* dst_gpu_tensor,
                    StatusCallback done);


    };
}//namespace dlxnet

#endif
