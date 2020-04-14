#include "dlxnet/core/common_runtime/bfc_allocator.h"

namespace dlxnet{
    BFCAllocator::BFCAllocator(SubAllocator* sub_allocator, size_t total_memory,
            bool allow_growth, const string& name,
            bool garbage_collection)
        : garbage_collection_(garbage_collection),
        sub_allocator_(sub_allocator),
        name_(name){
            // Allocate the requested amount of memory.
            memory_limit_ = total_memory;
            stats_.bytes_limit = static_cast<int64>(total_memory);
        }
    BFCAllocator::~BFCAllocator() {}
    void* BFCAllocator::AllocateRaw(size_t unused_alignment, size_t num_bytes,
            const AllocationAttributes& allocation_attr) {
        VLOG(1) << "AllocateRaw " << Name() << "  " << num_bytes;

    }

    void BFCAllocator::DeallocateRaw(void* ptr) {
        VLOG(1) << "DeallocateRaw " << Name() << " "
            << (ptr ? RequestedSize(ptr) : 0);
    }
    bool BFCAllocator::TracksAllocationSizes() const { return true; }

    size_t BFCAllocator::RequestedSize(const void* ptr) const {
        CHECK(ptr);
        mutex_lock l(lock_);
        return 0;
    }

    size_t BFCAllocator::AllocatedSize(const void* ptr) const {
        mutex_lock l(lock_);
        return 0;
    }

    int64 BFCAllocator::AllocationId(const void* ptr) const {
        mutex_lock l(lock_);
        return 0;
    }

    absl::optional<AllocatorStats> BFCAllocator::GetStats() {
        mutex_lock l(lock_);
        return stats_;
    }

    void BFCAllocator::ClearStats() {
        mutex_lock l(lock_);
        stats_.num_allocs = 0;
        stats_.peak_bytes_in_use = stats_.bytes_in_use;
        stats_.largest_alloc_size = 0;
    }
}//namespace dlxnet
