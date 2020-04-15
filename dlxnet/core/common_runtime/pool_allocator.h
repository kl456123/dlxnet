#ifndef DLXNET_CORE_COMMON_RUNTIME_POOL_ALLOCATOR_H_
#define DLXNET_CORE_COMMON_RUNTIME_POOL_ALLOCATOR_H_
#include <memory>

#include "dlxnet/core/framework/allocator.h"
#include "dlxnet/core/platform/macros.h"
#include "dlxnet/core/platform/mutex.h"
#include "dlxnet/core/platform/thread_annotations.h"

namespace dlxnet{
    // Size-limited pool of memory buffers obtained from a SubAllocator
    // instance.  Pool eviction policy is LRU.
    class PoolAllocator: public Allocator{
        public:
            // "pool_size_limit" is the maximum number of returned, re-usable
            // memory buffers to keep in the pool.  If pool_size_limit == 0, the
            // pool is effectively a thin wrapper around the allocator.
            // If "auto_resize" is true, then the pool_size_limit will gradually
            // be raised so that deallocations happen very rarely, if at all.
            // Transitory start-up objects may deallocate, but the long-term
            // working-set should not. Auto-resizing can raise pool_size_limit
            // but will never lower it.
            // "allocator" is the object that performs the underlying memory
            // malloc/free operations.  This object takes ownership of allocator.
            PoolAllocator(size_t pool_size_limit, bool auto_resize,
                    SubAllocator* allocator, string name);
            ~PoolAllocator() override;

            string Name() override { return name_; }

            void* AllocateRaw(size_t alignment, size_t num_bytes) override;

            void DeallocateRaw(void* ptr) override;

            // Current size limit.
            size_t size_limit() const NO_THREAD_SAFETY_ANALYSIS {
                return pool_size_limit_;
            }
        private:
            const string name_;
            const bool has_size_limit_;
            const bool auto_resize_;
            size_t pool_size_limit_;
            std::unique_ptr<SubAllocator> allocator_;
            mutex mutex_;
    };


    // basic sub allocator for cpu
    class BasicCPUAllocator : public SubAllocator {
        public:
            BasicCPUAllocator(int numa_node, const std::vector<Visitor>& alloc_visitors,
                    const std::vector<Visitor>& free_visitors)
                : SubAllocator(alloc_visitors, free_visitors), numa_node_(numa_node) {}

            ~BasicCPUAllocator() override {}

            void* Alloc(size_t alignment, size_t num_bytes) override;

            void Free(void* ptr, size_t num_bytes) override;

        private:
            int numa_node_;

            TF_DISALLOW_COPY_AND_ASSIGN(BasicCPUAllocator);
    };
}//namespace dlxnet

#endif
