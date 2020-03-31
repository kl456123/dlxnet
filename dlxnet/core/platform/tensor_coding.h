#ifndef DLXNET_CORE_PLATFORM_TENSOR_CODING_H_
#define DLXNET_CORE_PLATFORM_TENSOR_CODING_H_
#include "dlxnet/core/platform/platform.h"
#include "dlxnet/core/platform/types.h"
#include "dlxnet/core/platform/stringpiece.h"
#include "dlxnet/core/lib/core/refcount.h"

namespace dlxnet{
    namespace port{
        void AssignRefCounted(StringPiece src, core::RefCounted* obj, string* dst);
        void CopyFromArray(string* s, const char* base, size_t bytes);
        inline void CopyToArray(const string& src, char* dst){
            memcpy(dst, src.data(), src.size());
        }

        void EncodeStringList(const tstring* strings, int64 n, string* out);

        bool DecodeStringList(const string& src, tstring* strings, int64 n);
    }
}


#endif

