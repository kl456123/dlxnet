#ifndef DLXNET_CORE_COMMON_RUNTIME_BFC_ALLOCATOR_H_
#define DLXNET_CORE_COMMON_RUNTIME_BFC_ALLOCATOR_H_
#include "dlxnet/core/platform/mutex.h"
#include "dlxnet/core/framework/allocator.h"
#include "absl/container/flat_hash_set.h"

namespace dlxnet{
    // A memory allocator that implements a 'best-fit with coalescing'
    // algorithm.  This is essentially a very simple version of Doug Lea's
    // malloc (dlmalloc).
    //
    // The goal of this allocator is to support defragmentation via
    // coalescing.  One assumption we make is that the process using this
    // allocator owns pretty much all of the memory, and that nearly
    // all requests to allocate memory go through this interface.
    class BFCAllocator : public Allocator {
        public:
            // Takes ownership of sub_allocator.
            BFCAllocator(SubAllocator* sub_allocator, size_t total_memory,
                    bool allow_growth, const string& name,
                    bool garbage_collection = false);
            ~BFCAllocator() override;

            string Name() override { return name_; }

            void* AllocateRaw(size_t alignment, size_t num_bytes) override {
                return AllocateRaw(alignment, num_bytes, AllocationAttributes());
            }

            void* AllocateRaw(size_t alignment, size_t num_bytes,
                    const AllocationAttributes& allocation_attr) override;

            void DeallocateRaw(void* ptr) override;

            bool TracksAllocationSizes() const override;
            size_t RequestedSize(const void* ptr) const override;

            size_t AllocatedSize(const void* ptr) const override;

            int64 AllocationId(const void* ptr) const override;

            absl::optional<AllocatorStats> GetStats() override;

            void ClearStats() override;
        private:
            struct Bin;

            // Structures immutable after construction
            size_t memory_limit_ = 0;

            // Whether the allocator will deallocate free regions to avoid OOM due to
            // memory fragmentation.
            bool garbage_collection_;
            std::unique_ptr<SubAllocator> sub_allocator_;

            // Structures mutable after construction
            mutable mutex lock_;
            string name_;
            // Stats.
            AllocatorStats stats_ GUARDED_BY(lock_);
            TF_DISALLOW_COPY_AND_ASSIGN(BFCAllocator);
    };
}//namespace dlxnet

#endif
