#ifndef DLXNET_CC_OPS_CONST_OP_H_
#define DLXNET_CC_OPS_CONST_OP_H_
#include "dlxnet/cc/framework/scope.h"
#include "dlxnet/cc/framework/ops.h"
#include "dlxnet/core/framework/tensor.pb.h"

namespace dlxnet{
    namespace ops{
        // init from initializer_list
        Output Const(const Scope& scope, const Input& input);
        // init from proto
        Output ConstFromProto(const Scope& scope, const TensorProto& proto);
    }
}


#endif
