#include "dlxnet/core/platform/logging.h"
#include "dlxnet/core/platform/types.h"
#include "dlxnet/core/platform/stream_executor.h"
#include "dlxnet/core/common_runtime/gpu/gpu_init.h"


namespace dlxnet{
    Status ValidateGPUMachineManager() {
        return se::MultiPlatformManager::PlatformWithName(GpuPlatformName()).status();
    }

    se::Platform* GPUMachineManager() {
        auto result = se::MultiPlatformManager::PlatformWithName(GpuPlatformName());
        if (!result.ok()) {
            LOG(FATAL) << "Could not find Platform with name " << GpuPlatformName();
            return nullptr;
        }

        return result.ValueOrDie();
    }

    string GpuPlatformName() {
        // This function will return "CUDA" even when building TF without GPU support
        // This is done to preserve existing functionality
        return "OpenCL";
    }
}// namespace dlxnet
