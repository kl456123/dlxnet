#include "dlxnet/core/graph/tensor_id.h"


namespace dlxnet{
    TensorId ParseTensorName(const string& name){
        const char* base = name.data();
        const char* p = base+name.size()-1;
        //decode number
        unsigned int index = 0;
        unsigned int mul = 1;
        while(p>base&&(*p>='0'&&*p<='9')){
            index+=((*p-'0')*mul);
            mul*=10;
            p--;
        }
        // check ':' or not
        if(p>base&& *p==':'&&mul>1){
            id.first = string(base, p-base);
            id.second = index;
        }else{
            // even name ending with ':'
            id.first = name;
            id.second = 0;
        }
    }
}
