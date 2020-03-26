#ifndef DLXNET_CC_OPS_CONST_OP_H_
#define DLXNET_CC_OPS_CONST_OP_H_
#include "dlxnet/cc/framework/scope.h"
#include "dlxnet/cc/framework/ops.h"
#include "dlxnet/core/framework/tensor.pb.h"
#include "dlxnet/core/graph/node_builder.h"

namespace dlxnet{
    namespace ops{
        // init from initializer_list
        // const refers to any concrete value like tensor or tensor proto
        Node* Const(const Scope& scope, const Input::Initializer& val);

        // init from proto
        Node* ConstFromProto(const Scope& scope, const TensorProto& proto);

        // Construct NodeOut struct
        NodeBuilder::NodeOut AsNodeOut(const Scope& scope, const Input& inp);


        // from scalar and list type
        template<typename T>
            Node* Const(const Scope& scope, const T& value, const TensorShape shape){
                // convert to initializer first
                return Const(scope, Input::Initializer(value, shape));
            }

        template <typename T>
            Node* Const(const Scope& scope, const std::initializer_list<T>& v,
                    const TensorShape shape) {
                return Const(scope, Input::Initializer(v, shape));
            }
    }
}


#endif
