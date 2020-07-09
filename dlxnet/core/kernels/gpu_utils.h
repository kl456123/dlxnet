#ifndef DLXNET_CORE_KERNELS_GPU_UTILS_H_
#define DLXNET_CORE_KERNELS_GPU_UTILS_H_

#include "dlxnet/core/platform/logging.h"
#include "dlxnet/core/platform/stream_executor.h"

namespace dlxnet{
    template <typename T>
        inline se::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory, uint64 size) {
            se::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory), size * sizeof(T));
            se::DeviceMemory<T> typed(wrapped);
            return typed;
        }
} // namespace dlxnet


#endif
