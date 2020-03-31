#include "dlxnet/core/lib/core/coding.h"

namespace dlxnet{
    namespace core{
        void PutVarint32(string* dst, uint32 v){
            char buf[5];
            char* ptr = EncodeVarint32(buf, v);
            dst->append(buf, ptr-buf);
        }

        char* EncodeVarint32(char* dst, uint32 v){
            // the least significance bit is used to check there is any bits coming
            // Operate on characters as unsigneds
            unsigned char* ptr = reinterpret_cast<unsigned char*>(dst);
            static const int B = 128;
            if (v < (1 << 7)) {
                *(ptr++) = v;
            } else if (v < (1 << 14)) {
                *(ptr++) = v | B;
                *(ptr++) = v >> 7;
            } else if (v < (1 << 21)) {
                *(ptr++) = v | B;
                *(ptr++) = (v >> 7) | B;
                *(ptr++) = v >> 14;
            } else if (v < (1 << 28)) {
                *(ptr++) = v | B;
                *(ptr++) = (v >> 7) | B;
                *(ptr++) = (v >> 14) | B;
                *(ptr++) = v >> 21;
            } else {
                *(ptr++) = v | B;
                *(ptr++) = (v >> 7) | B;
                *(ptr++) = (v >> 14) | B;
                *(ptr++) = (v >> 21) | B;
                *(ptr++) = v >> 28;
            }
            return reinterpret_cast<char*>(ptr);

        }

        const char* GetVarint32Ptr(const char* p, const char* limit, uint32* value) {
            if (p < limit) {
                uint32 result = *(reinterpret_cast<const unsigned char*>(p));
                if ((result & 128) == 0) {
                    *value = result;
                    return p + 1;
                }
            }
            return GetVarint32PtrFallback(p, limit, value);
        }

        const char* GetVarint32PtrFallback(const char* p, const char* limit,
                uint32* value) {
            uint32 result = 0;
            for (uint32 shift = 0; shift <= 28 && p < limit; shift += 7) {
                uint32 byte = *(reinterpret_cast<const unsigned char*>(p));
                p++;
                if (byte & 128) {
                    // More bytes are present
                    result |= ((byte & 127) << shift);
                } else {
                    result |= (byte << shift);
                    *value = result;
                    return reinterpret_cast<const char*>(p);
                }
            }
            return nullptr;
        }

        bool GetVarint32(StringPiece* input, uint32* value) {
            const char* p = input->data();
            const char* limit = p + input->size();
            const char* q = GetVarint32Ptr(p, limit, value);
            if (q == nullptr) {
                return false;
            } else {
                *input = StringPiece(q, limit - q);
                return true;
            }
        }
    }
}
