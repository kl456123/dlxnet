#include "dlxnet/cc/ops/const_op.h"


namespace dlxnet{
    namespace ops{
        // just a function rather than a class
        Node* Const(const Scope& scope, const Input& input, const TensorProto&value, DataType dtype){
            Node* ret;
            Graph* graph = scope.graph();
            const string unique_name = scope.GetUniqueNameForOp("Const");
            auto builder = NodeBuilder(unique_name, "Const")
                .Attr("value", value)
                .Attr("dtype", dtype);

            scope.UpdateBuilder(&builder);
            scope.UpdateStatus(builder.Finalize(graph, &ret));

            if(!scope.ok())return nullptr;
            // shape inference after construct
            scope.UpdateStatus(scope.DoShapeInference(ret));
            if(!scope.ok())return nullptr;

            return ret;
        }
        Output ConstFromProto(const Scope& scope, const TensorProto& proto){
        }

        NodeBuilder::NodeOut AsNodeOut(const Scope& scope, const Input& inp){
            return NodeOut();
        }
    }
}
