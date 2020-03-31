#ifndef DLXNET_CORE_LIB_CORE_CODING_H_
#define DLXNET_CORE_LIB_CORE_CODING_H_
#include "dlxnet/core/lib/stringpiece.h"
#include "dlxnet/core/platform/types.h"

namespace dlxnet{
    namespace core{
        void PutVarint32(string* dst, uint32 v);
        bool GetVarint32(StringPiece* input, uint32* value);
        const char* GetVarint32Ptr(const char* p, const char* limit, uint32* value);
        const char* GetVarint32PtrFallback(const char* p, const char* limit,
                uint32* value);
        char* EncodeVarint32(char* dst, uint32 v);
    }
}


#endif
