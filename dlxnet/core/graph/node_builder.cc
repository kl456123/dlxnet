#include "dlxnet/core/graph/node_builder.h"

namespace dlxnet{
    // NodeOut
    NodeBuilder::NodeOut::NodeOut()
        : node(nullptr), index(0), dt(DT_FLOAT) {}
    NodeBuilder::NodeOut::NodeOut(Node* n, int32 i)
        : node(n), name(node!=nullptr?node->name():""), index(i){
            GetOutputType(node, i, &dt);
        }

    NodeBuilder::NodeBuilder(StringPiece name, StringPiece op_name,
            const OpRegistryInterface* op_registry)
        :def_builder_(name, op_name, op_registry){}
    NodeBuilder::NodeBuilder(StringPiece name, const OpDef* op_def)
        :def_builder_(name, op_def){
        }

    NodeBuilder::NodeBuilder(const NodeDefBuilder& def_builder)
        : def_builder_(def_builder) {}

    NodeBuilder& NodeBuilder::Input(Node* src_node, int src_index){
        inputs_.emplace_back(src_node, src_index);
        // get data type
        DataType dt;
        if(GetOutputType(src_node, src_index, &dt)){
            def_builder_.Input(src_node->name(), src_index, dt);
        }
        return *this;
    }

    NodeBuilder& NodeBuilder::Input(gtl::ArraySlice<NodeOut> src_list) {
        // std::vector<NodeDefBuilder::NodeOut> srcs;
        // srcs.reserve(src_list.size());
        // for (const auto& node_out : src_list) {
            // srcs.emplace_back(node_out.name, node_out.index, node_out.dt);
            // inputs_.emplace_back(node_out.node, node_out.index);
        // }
        // def_builder_.Input(gtl::ArraySlice<NodeDefBuilder::NodeOut>(srcs));
        return *this;
    }

    NodeBuilder& NodeBuilder::Device(StringPiece device_spec) {
        def_builder_.Device(device_spec);
        return *this;
    }

    NodeBuilder& NodeBuilder::AssignedDevice(StringPiece device) {
        assigned_device_ = string(device);
        return *this;
    }

    NodeBuilder& NodeBuilder::Input(NodeOut src){
        inputs_.emplace_back(src.node, src.index);
        def_builder_.Input(src.name, src.index, src.dt);
        return *this;
    }


    Status NodeBuilder::Finalize(Graph* graph, Node** created_node){
        // clear first
        if(created_node!=nullptr)*created_node = nullptr;
        if(!errors_.empty()){
            return errors::InvalidArgument(absl::StrJoin(errors_, "\n"));
        }
        // build node_def using def_builder
        NodeDef node_def;
        // check parse error
        TF_RETURN_IF_ERROR(def_builder_.Finalize(&node_def));
        // check semantic error, node_def should be consistent with op_def
        TF_RETURN_IF_ERROR(ValidateNodeDef(node_def, def_builder_.op_def()));

        Status status;
        Node* node = graph->AddNode(std::move(node_def), &status);
        if(!status.ok()){
            return status;
        }

        node->set_assigned_device_name(assigned_device_);

        // add edge
        for(int i=0;i<inputs_.size();i++){
            if(inputs_[i].node!=nullptr){
                graph->AddEdge(inputs_[i].node, inputs_[i].index, node, i);
            }
        }
        if(created_node!=nullptr) *created_node = node;
        return Status::OK();
    }

    /*static*/ bool NodeBuilder::GetOutputType(const Node* node, int i, DataType* dt) {
        if(node!=nullptr && i>=0&&i<node->num_outputs()){
            *dt = node->output_type(i);
            return true;
        }else{
            *dt = DT_FLOAT;
            return false;
        }
    }
} // namespace dlxnet
