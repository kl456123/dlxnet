#include "dlxnet/core/common_runtime/pool_allocator.h"
#include "dlxnet/core/platform/numa.h"
#include "dlxnet/core/platform/logging.h"
#include "dlxnet/core/platform/mem.h"

namespace dlxnet{
    PoolAllocator::PoolAllocator(size_t pool_size_limit, bool auto_resize,
            SubAllocator* allocator, string name)
        : name_(std::move(name)),
        has_size_limit_(pool_size_limit > 0),
        auto_resize_(auto_resize),
        pool_size_limit_(pool_size_limit),
        allocator_(allocator){
            if (auto_resize) {
                CHECK_LT(size_t{0}, pool_size_limit)
                    << "size limit must be > 0 if auto_resize is true.";
            }
        }

    PoolAllocator::~PoolAllocator() {}

    void PoolAllocator::DeallocateRaw(void* ptr) {
        allocator_->Free(ptr, 0);
    }

    void* PoolAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
        void* ptr = allocator_->Alloc(alignment, num_bytes);
    }

    void* BasicCPUAllocator::Alloc(size_t alignment, size_t num_bytes) {
        void* ptr = nullptr;
        if (num_bytes > 0) {
            if (numa_node_ == port::kNUMANoAffinity) {
                ptr = port::AlignedMalloc(num_bytes, static_cast<int>(alignment));
            } else {
                ptr =
                    port::NUMAMalloc(numa_node_, num_bytes, static_cast<int>(alignment));
            }
            VisitAlloc(ptr, numa_node_, num_bytes);
        }
        return ptr;
    }

    void BasicCPUAllocator::Free(void* ptr, size_t num_bytes) {
        if (num_bytes > 0) {
            VisitFree(ptr, numa_node_, num_bytes);
            if (numa_node_ == port::kNUMANoAffinity) {
                port::AlignedFree(ptr);
            } else {
                port::NUMAFree(ptr, num_bytes);
            }
        }
    }
}//namespace dlxnet
