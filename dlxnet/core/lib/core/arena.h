#ifndef DLXNET_CORE_LIB_CORE_ARENA_H_
#define DLXNET_CORE_LIB_CORE_ARENA_H_
#include <cstddef>

#include "dlxnet/core/framework/allocator.h"


namespace dlxnet{

    namespace core{
        class Arena{
            public:
                explicit Arena(const size_t block_size);
                ~Arena();

                void* Alloc(const size_t size);
                void* AllocAligned(const size_t size, const size_t alignment){
                    return nullptr;}

            private:
                Allocator* underlaying_allocator_;
        };
    }
}


#endif
