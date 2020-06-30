#ifndef DLXNET_CORE_COMMON_RUNTIME_GPU_GPU_ID_UTILS_H_
#define DLXNET_CORE_COMMON_RUNTIME_GPU_GPU_ID_UTILS_H_
#include "dlxnet/core/common_runtime/gpu/gpu_init.h"
#include "dlxnet/core/common_runtime/gpu/gpu_id.h"
#include "dlxnet/core/common_runtime/gpu/gpu_id_manager.h"


namespace dlxnet{
    // Utility methods for translation between Tensorflow GPU ids and platform GPU
    // ids.
    class GpuIdUtil {
        public:
            // Convenient methods for getting the associated executor given a TfGpuId or
            // PlatformGpuId.
            static se::port::StatusOr<se::StreamExecutor*> ExecutorForPlatformGpuId(
                    se::Platform* gpu_manager, PlatformGpuId platform_gpu_id) {
                return gpu_manager->ExecutorForDevice(platform_gpu_id);
            }
            static se::port::StatusOr<se::StreamExecutor*> ExecutorForPlatformGpuId(
                    PlatformGpuId platform_gpu_id) {
                return ExecutorForPlatformGpuId(GPUMachineManager(), platform_gpu_id);
            }
            static se::port::StatusOr<se::StreamExecutor*> ExecutorForTfGpuId(
                    TfGpuId tf_gpu_id) {
                PlatformGpuId platform_gpu_id;
                TF_RETURN_IF_ERROR(
                        GpuIdManager::TfToPlatformGpuId(tf_gpu_id, &platform_gpu_id));
                return ExecutorForPlatformGpuId(platform_gpu_id);
            }

            // Verify that the platform_gpu_id associated with a TfGpuId is legitimate.
            static void CheckValidTfGpuId(TfGpuId tf_gpu_id) {
                PlatformGpuId platform_gpu_id;
                TF_CHECK_OK(GpuIdManager::TfToPlatformGpuId(tf_gpu_id, &platform_gpu_id));
                const int visible_device_count = GPUMachineManager()->VisibleDeviceCount();
                CHECK_LT(platform_gpu_id, visible_device_count)
                    << "platform_gpu_id is outside discovered device range."
                    << " TF GPU id: " << tf_gpu_id
                    << " platform GPU id: " << platform_gpu_id
                    << " visible device count: " << visible_device_count;
            }
    };
} // namespace dlxnet


#endif
