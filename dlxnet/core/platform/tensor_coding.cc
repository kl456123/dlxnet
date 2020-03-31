#include <vector>

#include "dlxnet/core/platform/tensor_coding.h"
#include "dlxnet/core/lib/core/coding.h"

namespace dlxnet{
    namespace port{
        void AssignRefCounted(StringPiece src, core::RefCounted* obj, string* out){
            out->assign(src.data(), src.size());
        }
        void CopyFromArray(string* s, const char* base, size_t bytes){
            s->assign(base, bytes);
        }

        // tstring
        void EncodeStringList(const tstring* strings, int64 n, string* out){
            out->clear();
            for(int i=0;i<n;++i){
                core::PutVarint32(out, strings[i].size());
            }
            for(int i=0;i<n;++i){
                out->append(strings[i]);
            }
        }

        bool DecodeStringList(const string& src, tstring* strings, int64 n){
            std::vector<uint32> sizes(n);
            StringPiece reader(src);
            int64 tot = 0;
            // decode size first
            for(auto& v:sizes){
                if(!core::GetVarint32(&reader, &v))return false;
                tot+=v;
            }
            // check size
            if(tot!=static_cast<int64>(reader.size())){
                return false;
            }

            tstring* data = strings;
            for(int i=0;i<n;++i,++data){
                auto size = sizes[i];
                if(size>reader.size()){
                    return false;
                }
                data->assign(reader.data(), size);
                reader.remove_prefix(size);
            }
            return true;
        }

    }
}
