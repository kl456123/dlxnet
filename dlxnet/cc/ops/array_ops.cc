#include "dlxnet/cc/ops/array_ops.h"

namespace dlxnet{
    namespace ops{
        MatMul::MatMul(const ::dlxnet::Scope& scope, ::dlxnet::Input a,
                ::dlxnet::Input b)
            :MatMul(scope, a, b, MatMul::Attrs()){
            }
        MatMul::MatMul(const ::dlxnet::Scope& scope, ::dlxnet::Input a,
                ::dlxnet::Input b, const MatMul::Attrs& attrs){
            if(!scope.ok())return ;
            auto _a = ::dlxnet::ops::AsNodeOut(scope, a);
            auto _b = ::dlxnet::ops::AsNodeOut(scope, b);
            const string op_name = "MatMul";
            const string unique_name = scope.GetUniqueNameForOp(op_name);
            auto builder = NodeBuilder(unique_name, op_name)
                .Input(_a)
                .Input(_b);

            if(attrs.init_transpose_a){
                builder.Attr("transpose_a", attrs.transpose_a);
            }

            if(attrs.init_transpose_b){
                builder.Attr("transpose_b", attrs.transpose_b);
            }
            scope.UpdateBuilder(&builder);
            ::dlxnet::Node* ret;
            scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
            if(!scope.ok())return;
            scope.UpdateStatus(scope.DoShapeInference(ret));
            this->product = ret;
        }
    }
}
