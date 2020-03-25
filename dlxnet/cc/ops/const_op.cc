#include "dlxnet/cc/ops/const_op.h"


namespace dlxnet{
    namespace ops{
        namespace{
            template<typename T>
                Node* ConstHelper(const Scope& scope, const T&value, DataType dtype){
                    if(!scope.ok())return nullptr;
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
        }// namespace
        // just a function rather than a class

        Node* Const(const Scope& scope, const Input::Initializer& val){
            if(!val.status.ok()){
                scope.UpdateStatus(val.status);
                return nullptr;
            }
            return ConstHelper(scope, val.tensor, val.tensor.dtype());
        }
        Node* Const(const Scope& scope, const TensorProto& val){
            return ConstHelper(scope, val, val.dtype());
        }
        Output ConstFromProto(const Scope& scope, const TensorProto& proto){
        }

        // convert from input to node_out used in node builder
        NodeBuilder::NodeOut AsNodeOut(const Scope& scope, const Input& inp){
            // first check status of conversion to input
            if(!inp.status().ok()){
                scope.UpdateStatus(inp.status());
                return NodeBuilder::NodeOut(inp.node(), inp.index());
            }
            if(inp.node()){
                return NodeBuilder::NodeOut(inp.node(), inp.index());
            }

            auto transformed = Input{
                Const(scope.NewSubScope("Const"), Input::Initializer())};
            return NodeBuilder::NodeOut(transformed.node(), transformed.index());
        }
    }
}
