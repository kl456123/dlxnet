#ifndef DLXNET_CC_OPS_ARRAY_OPS_H_
#define DLXNET_CC_OPS_ARRAY_OPS_H_
/*
 *
 * */
/// this file is only used for supporting c++ api,
/// it can be generated automatically. But here we just
/// write it as a example
#include "dlxnet/cc/framework/scope.h"
#include "dlxnet/cc/framework/ops.h"
#include "dlxnet/cc/ops/const_op.h"

namespace dlxnet{
    namespace ops{
        class MatMul{
            public:
                struct Attrs{
                    // it is copyed too many times
                    TF_MUST_USE_RESULT Attrs TransposeA(bool x){
                        Attrs ret = *this;
                        ret.transpose_a = x;
                        return ret;
                    }
                    TF_MUST_USE_RESULT Attrs TransposeB(bool x){
                        Attrs ret = *this;
                        ret.transpose_b = x;
                        return ret;
                    }
                    bool transpose_a=false;
                    bool transpose_b = false;
                };
                MatMul(const ::dlxnet::Scope& scope, ::dlxnet::Input a,
                ::dlxnet::Input b, const MatMul::Attrs& attrs);
                MatMul(const ::dlxnet::Scope& scope, ::dlxnet::Input a,
                ::dlxnet::Input b);
                operator ::dlxnet::Input()const{return product;}
                ::dlxnet::Node* product;

                // used for client user
                static Attrs TransposeA(bool x){
                    return Attrs().TransposeA(x);
                }
                static Attrs TransposeB(bool x){
                    return Attrs().TransposeB(x);
                }
        };
    }
}


#endif
