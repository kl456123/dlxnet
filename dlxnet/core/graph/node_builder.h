#ifndef DLXNET_CORE_GRAPH_NODE_BUILDER_H_
#define DLXNET_CORE_GRAPH_NODE_BUILDER_H_
#include "dlxnet/core/platform/types.h"
#include "dlxnet/core/lib/status.h"
#include "dlxnet/core/graph/graph.h"
#include "dlxnet/core/framework/types.pb.h"
#include "dlxnet/core/framework/op_def.pb.h"
#include "dlxnet/core/framework/node_def_builder.h"

namespace dlxnet{
    // helper class to build node(add node to graph)
    // It using NodeDefBuilder internally
    class NodeBuilder{
        public:
            // construct all in one struct
            struct NodeOut{
                NodeOut(Node* n, int32 i=0);
                NodeOut();
                Node* node;
                string name;
                int32 index;
                DataType dt;
            };
            // Specify the name and the Op (either via an OpDef or the name of
            // the Op plus a registry) for the Node.  Other fields are
            // specified by calling the methods below.
            // REQUIRES: The OpDef must satisfy ValidateOpDef().
            NodeBuilder(StringPiece name, StringPiece op_name,
                    const OpRegistryInterface* op_registry = OpRegistry::Global());
            NodeBuilder(StringPiece name, const OpDef* op_def);
            NodeBuilder& Device(const string& string);
            NodeBuilder& Input(Node* src_node, int src_index=0);
            NodeBuilder& Input(NodeOut src);
            template<class T>
                NodeBuilder& Attr(StringPiece attr_name, T&& value);
            Status Finalize(Graph* graph, Node** created_node);

            // Set *dt and returns true if i is in range. Combines
            // SafeGetOutput() and AddIndexError().
            static bool GetOutputType(const Node* node, int i, DataType* dt);

        private:
            NodeDefBuilder def_builder_;
            std::vector<string> errors_;
            std::vector<NodeOut> inputs_;
    };



    // IMPLEMENTATION -------------------------------------------------------------

    template <class T>
        NodeBuilder& NodeBuilder::Attr(StringPiece attr_name, T&& value) {
            def_builder_.Attr(attr_name, std::forward<T>(value));
            return *this;
        }
}// namespace dlxnet


#endif
