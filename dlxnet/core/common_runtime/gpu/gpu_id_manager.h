#ifndef DLXNET_CORE_COMMON_RUNTIME_GPU_GPU_ID_MANAGER_H_
#define DLXNET_CORE_COMMON_RUNTIME_GPU_GPU_ID_MANAGER_H_
#include "dlxnet/core/common_runtime/gpu/gpu_id.h"
#include "dlxnet/core/lib/core/status.h"

namespace dlxnet{
    // Class that maintains a map from TfGpuId to PlatformGpuId, and manages the
    // translation between them.
    class GpuIdManager {
        public:
            // Adds a mapping from tf_gpu_id to platform_gpu_id.
            static Status InsertTfPlatformGpuIdPair(TfGpuId tf_gpu_id,
                    PlatformGpuId platform_gpu_id);

            // Gets the platform_gpu_id associated with tf_gpu_id. Returns OK if found.
            static Status TfToPlatformGpuId(TfGpuId tf_gpu_id,
                    PlatformGpuId* platform_gpu_id);
    };

}//namespace dlxnet
#endif
