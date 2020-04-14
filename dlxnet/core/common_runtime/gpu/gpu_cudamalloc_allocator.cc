#include "dlxnet/core/common_runtime/gpu/gpu_cudamalloc_allocator.h"

namespace dlxnet{
    GPUcudaMallocAllocator::GPUcudaMallocAllocator(Allocator* allocator,
            PlatformGpuId platform_gpu_id)
        : base_allocator_(allocator) {}

    GPUcudaMallocAllocator::~GPUcudaMallocAllocator() { delete base_allocator_; }

    void* GPUcudaMallocAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
#ifdef GOOGLE_CUDA
        // allocate with cudaMalloc
        // se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
        CUdeviceptr rv = 0;
        CUresult res = cuMemAlloc(&rv, num_bytes);
        if (res != CUDA_SUCCESS) {
            LOG(ERROR) << "cuMemAlloc failed to allocate " << num_bytes;
            return nullptr;
        }
        return reinterpret_cast<void*>(rv);
#else
        return nullptr;
#endif  // GOOGLE_CUDA
    }
    void GPUcudaMallocAllocator::DeallocateRaw(void* ptr) {
#ifdef GOOGLE_CUDA
        // free with cudaFree
        CUresult res = cuMemFree(reinterpret_cast<CUdeviceptr>(ptr));
        if (res != CUDA_SUCCESS) {
            LOG(ERROR) << "cuMemFree failed to free " << ptr;
        }
#endif  // GOOGLE_CUDA
    }

    absl::optional<AllocatorStats> GPUcudaMallocAllocator::GetStats() {
        return base_allocator_->GetStats();
    }

    bool GPUcudaMallocAllocator::TracksAllocationSizes() const { return false; }
}//namespace dlxnet
