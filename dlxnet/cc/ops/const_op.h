#ifndef DLXNET_CC_OPS_CONST_OP_H_
#define DLXNET_CC_OPS_CONST_OP_H_
#include "dlxnet/cc/framework/scope.h"
#include "dlxnet/cc/framework/ops.h"
#include "dlxnet/core/framework/tensor.pb.h"
#include "dlxnet/core/graph/node_builder.h"

namespace dlxnet{
    namespace ops{
        // init from initializer_list
        Node* Const(const Scope& scope, const Input& input, const TensorProto&value, DataType dtype);
        // init from proto
        Output ConstFromProto(const Scope& scope, const TensorProto& proto);

        // Construct NodeOut struct
        NodeBuilder::NodeOut AsNodeOut(const Scope& scope, const Input& inp);
    }
}


#endif
