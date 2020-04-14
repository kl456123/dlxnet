#ifndef DLXNET_CORE_COMMON_RUNTIME_GPU_GPU_CUDAMALLOC_ALLOCATOR_H_
#define DLXNET_CORE_COMMON_RUNTIME_GPU_GPU_CUDAMALLOC_ALLOCATOR_H_
#include "dlxnet/core/framework/allocator.h"
#include "dlxnet/core/common_runtime/gpu/gpu_id.h"
#include "dlxnet/core/platform/macros.h"

namespace dlxnet{
    // An allocator that wraps a GPU allocator and adds debugging
    // functionality that verifies that users do not write outside their
    // allocated memory.
    class GPUcudaMallocAllocator : public Allocator {
        public:
            explicit GPUcudaMallocAllocator(Allocator* allocator,
                    PlatformGpuId platform_gpu_id);
            ~GPUcudaMallocAllocator() override;
            string Name() override { return "gpu_debug"; }
            void* AllocateRaw(size_t alignment, size_t num_bytes) override;
            void DeallocateRaw(void* ptr) override;
            bool TracksAllocationSizes() const override;
            absl::optional<AllocatorStats> GetStats() override;

        private:
            Allocator* base_allocator_ = nullptr;  // owned

            TF_DISALLOW_COPY_AND_ASSIGN(GPUcudaMallocAllocator);
    };
}

#endif
