#ifndef DLXNET_CORE_COMMON_RUNTIME_GPU_GPU_MEM_ALLOCATOR_H_
#define DLXNET_CORE_COMMON_RUNTIME_GPU_GPU_MEM_ALLOCATOR_H_
#include "dlxnet/core/common_runtime/gpu/gpu_id.h"
#include "dlxnet/core/framework/allocator.h"
#include "dlxnet/core/framework/logging.h"

namespace dlxnet{
    // Suballocator for GPU memory.
    class GPUMemAllocator : public SubAllocator {
        public:
            // 'platform_gpu_id' refers to the ID of the GPU device within
            // the process and must reference a valid ID in the process.
            // Note: stream_exec cannot be null.
            explicit GPUMemAllocator(PlatformGpuId gpu_id, bool use_unified_memory,
                    const std::vector<Visitor>& alloc_visitors,
                    const std::vector<Visitor>& free_visitors)
                : SubAllocator(alloc_visitors, free_visitors),
                gpu_id_(gpu_id),
                use_unified_memory_(use_unified_memory) {
                    // CHECK(stream_exec_ != nullptr);
                }
            ~GPUMemAllocator() override {}

            void* Alloc(size_t alignment, size_t num_bytes) override {
                void* ptr = nullptr;
                if (num_bytes > 0) {
                    // if (use_unified_memory_) {
                    // ptr = stream_exec_->UnifiedMemoryAllocate(num_bytes);
                    // } else {
                    // ptr = stream_exec_->AllocateArray<char>(num_bytes).opaque();
                    // }
                    VisitAlloc(ptr, gpu_id_, num_bytes);
                }
                return ptr;
            }

            void Free(void* ptr, size_t num_bytes) override {
                if (ptr != nullptr) {
                    VisitFree(ptr, gpu_id_, num_bytes);
                    if (use_unified_memory_) {
                        // stream_exec_->UnifiedMemoryDeallocate(ptr);
                        // } else {
                        // se::DeviceMemoryBase gpu_ptr(ptr);
                        // stream_exec_->Deallocate(&gpu_ptr);
                }
                }
            }
        private:
            const PlatformGpuId gpu_id_;
            const bool use_unified_memory_ = false;


    };
}// namespace dlxnet

#endif

