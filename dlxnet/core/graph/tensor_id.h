#ifndef DLXNET_CORE_GRAPH_TENSOR_H_
#define DLXNET_CORE_GRAPH_TENSOR_H_

#include "dlxnet/core/platform/types.h"
#include "dlxnet/core/lib/strcat.h"

namespace dlxnet{
    struct TensorId{
        int index()const{ return second;}
        const string& node()const {return first;}
        int second;
        string first;
        string ToString()const{
            return strings::StrCat(first, ":", second);
        }

        struct Hasher{
            public:
                std::size_t operator()(const TensorId& x)const{
                    return Hash32(x.first.data(), x.first.size(),x.second);
                }
        };
    };

    TensorId ParseTensorName(const string& name);
}


#endif
