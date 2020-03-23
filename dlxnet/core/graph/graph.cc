#include "dlxnet/core/graph/graph.h"


namespace dlxnet{
    // Node implement
    struct NodeProperties {
        public:
            NodeProperties(const OpDef* op_def, NodeDef node_def,
                    const DataTypeSlice inputs, const DataTypeSlice outputs)
                : op_def(op_def),
                node_def(std::move(node_def)),
                input_types(inputs.begin(), inputs.end()),
                output_types(outputs.begin(), outputs.end()) {}

            const OpDef* op_def;  // not owned
            NodeDef node_def;
            const DataTypeVector input_types;
            const DataTypeVector output_types;
    };

    Node::NodeClass Node::GetNodeClassForOp(const string& ts) {
        return NC_OTHER;
    }

    void Node::Initialize(int id,
            std::shared_ptr<NodeProperties> props) {
        DCHECK_EQ(id_, -1);
        DCHECK(in_edges_.empty());
        DCHECK(out_edges_.empty());
        id_ = id;

        props_ = std::move(props);
        // Initialize the class_ based on the type string
        class_ = GetNodeClassForOp(props_->node_def.op());
    }

    string Node::DebugString() const {
        string ret = strings::StrCat("{name:'", name(), "' id:", id_);
        if (IsSource()) {
            strings::StrAppend(&ret, " source}");
        } else if (IsSink()) {
            strings::StrAppend(&ret, " sink}");
        } else {
            strings::StrAppend(&ret, " op device:");
            strings::StrAppend(&ret, "{", assigned_device_name(), "}");
            strings::StrAppend(&ret, " def:{", SummarizeNode(*this), "}}");
        }
        return ret;
    }

    Node::Node()
        : id_(-1),
        class_(NC_UNINITIALIZED),
        props_(nullptr),
        assigned_device_name_index_(0){}
    void Node::UpdateProperties(){}

    const string& Node::name() const { return props_->node_def.name(); }
    const string& Node::type_string() const { return props_->node_def.op(); }
    const NodeDef& Node::def() const { return props_->node_def; }
    const OpDef& Node::op_def() const { return *props_->op_def; }

    // inputs and outputs accessor
    int32 Node::num_inputs() const { return props_->input_types.size(); }
    DataType Node::input_type(int32 i) const { return props_->input_types[i]; }
    const DataTypeVector& Node::input_types() const { return props_->input_types; }

    int32 Node::num_outputs() const { return props_->output_types.size(); }
    DataType Node::output_type(int32 o) const { return props_->output_types[o]; }
    const DataTypeVector& Node::output_types() const {
        return props_->output_types;
    }

    AttrSlice Node::attrs() const { return AttrSlice(def()); }

    // Graph implementation
    Graph::Graph(const OpRegistryInterface* ops)
        :ops_(OpRegistry::Global()),
        arena_(8 << 10 /* 8kB */){
        }
    Graph::~Graph(){
        // Manually call the destructors for all the Nodes we constructed using
        // placement new.
        for (Node* node : nodes_) {
            if (node != nullptr) {
                node->~Node();
            }
        }
        for (Node* node : free_nodes_) {
            node->~Node();
        }
        // Edges have no destructor, and we arena-allocated them, so no need to
        // destroy them.
    }
    Node* Graph::AllocateNode(std::shared_ptr<NodeProperties> props){
        Node* node;
        // check any node free exist
        if(free_nodes_.empty()){
            node = new (arena_.Alloc(sizeof(Node))) Node;  // placement new
        }else{
            node = free_nodes_.back();
        }

        // initialize node here
        const int id = nodes_.size();
        node->graph_ = this;
        node->Initialize(id, std::move(props));
        nodes_.push_back(node);
        ++num_nodes_;
        return node;
    }

    Node* Graph::AddNode(NodeDef node_def, Status* status) {
        const OpRegistrationData* op_reg_data;
        status->Update(ops_->LookUp(node_def.op(), &op_reg_data));
        if(!status->ok())return nullptr;

        // prepare input types and output types
        DataTypeVector inputs;
        DataTypeVector outputs;
        status->Update(
                InOutTypesForNode(node_def, op_reg_data->op_def, &inputs, &outputs));
        if(!status->ok()){
            return nullptr;
        }

        Node* node = AllocateNode(
                std::make_shared<NodeProperties>(&op_reg_data->op_def,
                    std::move(node_def), inputs, outputs));

        return node;
    }

    const Edge* Graph::AddEdge(Node* source, int x, Node* dest, int y){
        // check any edge free exist
        Edge* e = nullptr;
        if (free_edges_.empty()) {
            e = new (arena_.Alloc(sizeof(Edge))) Edge;  // placement new
        } else {
            e = free_edges_.back();
            free_edges_.pop_back();
        }

        // populate edge, add it to src node and dst node,
        // then add it to graph
        e->id_ = edges_.size();
        e->src_ = source;
        e->dst_ = dest;
        e->src_output_ = x;
        e->dst_input_ = y;
        CHECK(source->out_edges_.insert(e).second);
        CHECK(dest->in_edges_.insert(e).second);
        edges_.push_back(e);
        ++num_edges_;
        return e;
    }
}