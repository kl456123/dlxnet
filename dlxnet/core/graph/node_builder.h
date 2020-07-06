#ifndef DLXNET_CORE_GRAPH_NODE_BUILDER_H_
#define DLXNET_CORE_GRAPH_NODE_BUILDER_H_
#include "dlxnet/core/platform/types.h"
#include "dlxnet/core/lib/core/status.h"
#include "dlxnet/core/graph/graph.h"
#include "dlxnet/core/framework/types.pb.h"
#include "dlxnet/core/framework/op_def.pb.h"
#include "dlxnet/core/framework/node_def_builder.h"
#include "dlxnet/core/lib/gtl/array_slice.h"
#include "dlxnet/core/lib/core/stringpiece.h"

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

            // Create a NodeBuilder from an existing NodeDefBuilder.
            NodeBuilder(const NodeDefBuilder& def_builder);

            // Sets the "requested device spec" in the NodeDef (not the
            // "assigned device" in the Node).
            NodeBuilder& Device(StringPiece device_spec);

            // Sets the device name in the "assigned device" field in tensorflow::Node.
            NodeBuilder& AssignedDevice(StringPiece device);

            // You must call one Input() function per input_arg in the Op,
            // *and in the same order as the input_args appear in the OpDef.*

            // For inputs that take a single tensor.
            NodeBuilder& Input(Node* src_node, int src_index = 0);
            NodeBuilder& Input(NodeOut src);

            // For inputs that take a list of tensors.
            NodeBuilder& Input(gtl::ArraySlice<NodeOut> src_list);

            // Set the value of an attr.  attr_name must match the name of one of
            // attrs defined by the Op, and value must have the corresponding type
            // (see SetAttrValue() in ../framework/attr_value_util.h for legal
            // types for value).  Note that attrs will be set automatically if
            // they can be determined by the inputs.
            template <class T>
                NodeBuilder& Attr(StringPiece attr_name, T&& value);
            template <class T>
                NodeBuilder& Attr(StringPiece attr_name, std::initializer_list<T> value);

            // Validates the described node and adds it to *graph, adding edges
            // for all (non-back) inputs.  If created_node is not nullptr,
            // *created_node will be set to the new node (or nullptr on error).
            // If `consume` is true, the builder state will be moved into `node_def`,
            // and the builder will be left in an undefined state.
            Status Finalize(Graph* graph, Node** created_node);



            // Accessors for the values set in the constructor.
            const string& node_name() const { return def_builder_.node_name(); }
            const OpDef& op_def() const { return def_builder_.op_def(); }

        private:
            // Set *dt and returns true if i is in range. Combines
            // SafeGetOutput() and AddIndexError().
            static bool GetOutputType(const Node* node, int i, DataType* dt);

            NodeDefBuilder def_builder_;
            std::vector<string> errors_;
            std::vector<NodeOut> inputs_;
            string assigned_device_;
    };



    // IMPLEMENTATION -------------------------------------------------------------

    template <class T>
        NodeBuilder& NodeBuilder::Attr(StringPiece attr_name, T&& value) {
            def_builder_.Attr(attr_name, std::forward<T>(value));
            return *this;
        }

    template <class T>
        NodeBuilder& NodeBuilder::Attr(StringPiece attr_name,
                std::initializer_list<T> value) {
            def_builder_.Attr(attr_name, value);
            return *this;
        }
}// namespace dlxnet


#endif
