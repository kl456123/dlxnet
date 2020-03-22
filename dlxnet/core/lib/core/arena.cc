#include "dlxnet/core/lib/core/arena.h"
#include "dlxnet/core/platform/numa.h"


namespace dlxnet{
    namespace core{
        Arena::Arena(const size_t block_size){
            underlaying_allocator_ = cpu_allocator(port::kNUMANoAffinity);
        }
        Arena::~Arena(){}
        void* Arena::Alloc(const size_t size){
            underlaying_allocator_->AllocateRaw(Allocator::kAllocatorAlignment, size);
        }
    }// namespace core

}// namespace dlxnet
