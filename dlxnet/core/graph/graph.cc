#include "dlxnet/core/graph/graph.h"
#include "dlxnet/core/lib/stringprintf.h"


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
            // Initialize the name interning table for assigned_device_name.
            device_names_.push_back("");
            DCHECK_EQ(0, InternDeviceName(""));
        }

    void Graph::RemoveEdge(const Edge* e) {
        TF_DCHECK_OK(IsValidNode(e->src_)) << e->src_->DebugString();
        TF_DCHECK_OK(IsValidNode(e->dst_)) << e->dst_->DebugString();
        CHECK_EQ(e->src_->out_edges_.erase(e), size_t{1});
        CHECK_EQ(e->dst_->in_edges_.erase(e), size_t{1});
        CHECK_EQ(e, edges_[e->id_]);
        CHECK_GT(num_edges_, 0);

        edges_[e->id_] = nullptr;
        free_edges_.push_back(const_cast<Edge*>(e));
        --num_edges_;
    }

    Status Graph::IsValidNode(const Node* node) const {
        if (node == nullptr) {
            return errors::InvalidArgument("Node is null");
        }
        const int id = node->id();
        if (id < 0) {
            return errors::InvalidArgument("node id ", id, " is less than zero");
        }
        if (static_cast<size_t>(id) >= nodes_.size()) {
            return errors::InvalidArgument(
                    "node id ", id, " is >= than number of nodes in graph ", nodes_.size());
        }
        if (nodes_[id] != node) {
            return errors::InvalidArgument("Node with id ", id,
                    " is different from the passed in node. "
                    "Does it belong to a different graph?");
        }
        return Status::OK();
    }

    int Graph::InternDeviceName(const string& device_name){
        if(device_name.empty()){
            // avoid lookup in table
            return 0;
        }

        int& index_cell = device_names_map_[device_name];
        if(index_cell>0){
            // exist in the table
            return index_cell;
        }

        // not exsit in the table
        index_cell = device_names_map_.size();
        device_names_.push_back(device_name);
        return index_cell;
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

    Node* Graph::CopyNode(const Node* node){
        Node* copy = AllocateNode(node->props_);
        copy->set_assigned_device_name(node->assigned_device_name());
        // Since the OpDef of a function may be owned by the Graph that owns 'node',
        // relookup the OpDef in the target graph. If it differs, then clone the
        // node properties with the updated OpDef.
        const OpDef* op_def;
        TF_CHECK_OK(ops_->LookUpOpDef(node->type_string(), &op_def));
        if (op_def != node->props_->op_def) {
            copy->props_->op_def = op_def;
        }
        return copy;
    }

    namespace {
        void AddInput(NodeDef* dst, StringPiece src_name, int src_slot){
            if(src_slot==0){
                // just omit index when it's zero
                dst->add_input(src_name.data(), src_name.size());
            }else{
                dst->add_input(strings::StrCat(src_name, ":", src_slot));
            }
        }
    }// namespace

    void Graph::ToGraphDef(GraphDef* graph_def) const{
        graph_def->Clear();
        graph_def->mutable_node()->Reserve(std::max(1, num_nodes()));

        std::vector<const Edge*> inputs;// construct once shared for all nodes
        for(auto id =0;id<num_node_ids();++id){
            const Node* node  = FindNodeId(id);
            if(node==nullptr)continue;
            NodeDef* node_def = graph_def->add_node();
            *node_def = node->def();

            // set assigned device name
            if(!node->assigned_device_name().empty()){
                node_def->set_device(node->assigned_device_name());
            }

            inputs.clear();
            inputs.resize(node->num_inputs(), nullptr);
            // fill inputs
            for(const Edge* edge: node->in_edges()){
                DCHECK(edge->dst_input() < inputs.size())
                    << "Edge " << edge->DebugString()
                    << " is overflowing the expected number of inputs ("
                    << node->num_inputs() << ") for node " << node->DebugString();
                CHECK(inputs[edge->dst_input()] == nullptr)
                    << "Edge " << edge->src()->name() << "->" << edge->dst()->name()
                    << " conflicts with pre-existing input edge "
                    << inputs[edge->dst_input()]->src()->name() << "->"
                    << inputs[edge->dst_input()]->dst()->name();

                inputs[edge->dst_input()] = edge;
            }

            // add each input to node_def
            node_def->clear_input();
            node_def->mutable_input()->Reserve(inputs.size());

            for(size_t i=0;i<inputs.size();++i){
                const Edge* edge = inputs[i];
                if(edge==nullptr){
                    node_def->add_input("");
                }else{
                    const Node* src = edge->src();
                    // handle input name format
                    AddInput(node_def, src->name(), edge->src_output());
                }

            }
        }

    }
    string Edge::DebugString() const {
        return strings::Printf("[id=%d %s:%d -> %s:%d]", id_, src_->name().c_str(),
                src_output_, dst_->name().c_str(), dst_input_);
    }
}// namespace dlxnet
