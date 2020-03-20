#include "dlxnet/core/graph/graph_constructor.h"
#include "dlxnet/core/graph/tensor_id.h"
#include "dlxnet/core/graph/graph_constructor.h"
#include "dlxnet/core/platform/macros.h"
#include "dlxnet/core/framework/op.h"
#include "dlxnet/core/framework/types.h"
#include "dlxnet/core/lib/status.h"
#include "dlxnet/core/platform/logging.h"
#include "dlxnet/core/lib/gtl/flatmap.h"
#include "dlxnet/core/common_runtime/shape_refiner.h"
#include "dlxnet/core/framework/node_def_util.h"


namespace dlxnet{

    namespace {
        class GraphConstructor{
            public:
                GraphConstructor(const GraphConstructorOptions& opts,
                        GraphDef&& gdef, Graph* g, ShapeRefiner* refiner)
                    :graph_def_(gdef),
                    g_(g),
                    refiner_(refiner),
                    opts_(opts){
                    }

                static Status Construct(const GraphConstructorOptions& opts,
                        GraphDef&& gdef, Graph* g, ShapeRefiner* refiner);
                Status TryImport() {
                    TF_RETURN_IF_ERROR(EnsureNoNameCollisions());
                    // TF_RETURN_IF_ERROR(ValidateInputMapAndControlDependencies());
                    TF_RETURN_IF_ERROR(BuildNodeIndex());
                    TF_RETURN_IF_ERROR(InitFromEdges());

                    // NOTE: Convert() invokes `consume_node_def()` on each node in the input
                    // graph, so `get_node_def()` is no longer usable once it is called.
                    TF_RETURN_IF_ERROR(Convert());

                    // TF_RETURN_IF_ERROR(AddBackEdges());
                    // TF_RETURN_IF_ERROR(UpdateVersionDef());
                    TF_RETURN_IF_ERROR(PopulateReturnTensors());
                    TF_RETURN_IF_ERROR(PopulateReturnNodes());
                    TF_RETURN_IF_ERROR(PopulateMissingUnusedInputMapKeys());
                    // UpdateUniquifiedColocationNames();
                    FixupSourceAndSinkEdges();
                    return Status::OK();
                }
                void Undo();
            private:
                void FixupSourceAndSinkEdges();

                Status EnsureNoNameCollisions();
                Status ValidateInputMapAndControlDependencies();
                Status BuildNodeIndex();
                Status InitFromEdges();
                Status Convert();
                // Status AddBackEdges();
                // Status UpdateVersionDef();
                Status PopulateReturnTensors();
                Status PopulateReturnNodes();
                Status PopulateMissingUnusedInputMapKeys();

                Status MakeNode(NodeDef&& node_def, Node** node);
                Status MakeEdge(Node* src, int output_index, Node* dst, int input_index);
                Status ValidateShape(Node* node);

                // Returns true if `name` already exists in `g_` (either as a node name or
                // prefix).
                bool NameExistsInGraph(StringPiece name);

                // Returns true if `name` already exists in the GraphDef being imported
                // (either as a node name or prefix).
                bool NameExistsInGraphDef(StringPiece name);

                // Returns a unique version of `original_name`, or `original_name` if it's
                // already unique in the graph.
                string FindUniqueName(StringPiece original_name);

                // Decrement pending count for users of `processed` and add the ones that now
                // have all of their pending inputs satisfied to `ready_`.
                void UpdatePendingCountAndReady(int processed);

                GraphDef graph_def_;
                std::vector<bool> is_consumed_;

                // Returns the number of nodes in the graph.
                virtual size_t node_def_count() const{ return graph_def_.node().size(); };
                // Returns the i^th node in the graph. Must not be called after
                // consume_node_def(i).
                const NodeDef& get_node_def(int i) const {
                    CHECK(!is_consumed_[i])
                        << "NodeDef " << i << " accessed after it was consumed.";
                    return graph_def_.node(i);
                }
                // Destructively reads the i^th node in the graph, avoiding a copy if
                // possible. After calling this method, the result of get_node_def(i) is
                // undefined.
                virtual NodeDef consume_node_def(int i){
                    CHECK(!is_consumed_[i]) << "NodeDef " << i << " consumed twice.";
                    is_consumed_[i] = true;
                    return std::move(*graph_def_.mutable_node(i));
                }

                // From constructor
                const GraphConstructorOptions opts_;
                Graph* g_;
                ShapeRefiner* refiner_;
                struct NodeInfo{
                    explicit NodeInfo(int i) : gdef_index(i), node(nullptr) {}
                    // Containers require that we have a default constructor.
                    NodeInfo() : NodeInfo(-1) {}
                    int gdef_index;
                    Node* node;  // nullptr until the NodeDef is converted to a Node.
                };
                gtl::FlatMap<StringPiece, NodeInfo, StringPieceHasher> gdef_nodes_;

                // Imported node names that have been uniquified. The key is the original
                // name, the value is the new unique name.
                gtl::FlatMap<string, string> uniquified_names_;

                // Index of NodeDefs in node_defs_ with all inputs already converted. We use a
                // (sorted) set so nodes are created in the order defined in the GraphDef.
                std::set<int> ready_;
                // Mapping between index within node_defs_ and the number of inputs that
                // still need to be converted.
                std::vector<int> pending_count_;
                // Mapping from node name to the existing node in g_.
                gtl::FlatMap<StringPiece, Node*, StringPieceHasher> existing_nodes_;

                struct InputInfo{
                    explicit InputInfo(const string& node_name, Node* n, int i)
                        :name(node_name), node(n), index(i){}
                    string name;
                    Node* node;
                    int index;
                };

                // Mapping between index within node_defs_ and the index within node_defs_ of
                // all nodes it outputs to.
                std::vector<gtl::InlinedVector<int, 4>> outputs_;
                TF_DISALLOW_COPY_AND_ASSIGN(GraphConstructor);
        };


        Status GraphConstructor::Construct(const GraphConstructorOptions& opts,
                GraphDef&& gdef, Graph* g, ShapeRefiner* refiner){
            GraphConstructor c(opts, std::move(gdef), g, refiner);
            const Status s = c.TryImport();
            if (!s.ok()) c.Undo();
            return s;
        }
        Status GraphConstructor::BuildNodeIndex(){
            // generate gdef_nodes_
            for(int n=0;n<node_def_count();n++){
                const NodeDef& node_def = get_node_def(n);
                if (!gdef_nodes_
                        .insert(std::make_pair(StringPiece(node_def.name()), NodeInfo(n)))
                        .second) {
                    return errors::InvalidArgument("Node '", node_def.name(),
                            "' is not unique");
                }
                // valid op type and device
                if (node_def.op().empty()) {
                    return errors::InvalidArgument("Node '", node_def.name(),
                            "' does not specify an operation");
                }
                if (node_def.device().empty()) {
                    return errors::InvalidArgument("Node '", node_def.name(),
                            "' is missing a device specification");
                }
            }
            return Status::OK();
        }
        Status GraphConstructor::InitFromEdges() {
            // generate pending_count_ and outputs_ for each node
            // Parse the inputs for each node.
            const int num_nodes = node_def_count();
            pending_count_.reserve(num_nodes);
            outputs_.resize(num_nodes);
            for (int n = 0; n < num_nodes; ++n) {
                const NodeDef& node_def = get_node_def(n);
                int pending_count = node_def.input_size();
                for (int i = 0; i < node_def.input_size(); ++i){
                    StringPiece input_name = node_def.input(i);
                    TensorId id(ParseTensorName(string(input_name)));

                    auto iter = gdef_nodes_.find(id.first);
                    if (iter == gdef_nodes_.end()) {
                        return errors::InvalidArgument("Node '", node_def.name(),
                                "': Unknown input node '",
                                node_def.input(i), "'");
                    }
                    outputs_[iter->second.gdef_index].push_back(n);
                }
                if (pending_count == 0) {
                    ready_.insert(n);
                }
                pending_count_.push_back(pending_count);
            }
            return Status::OK();
        }

        Status GraphConstructor::ValidateShape(Node* node) {
            if (!opts_.validate_shape) return Status::OK();
            TF_RETURN_IF_ERROR(refiner_->AddNode(node));
            // for some specifial output shape attr
            // auto* ic = refiner_->GetContext(node);
            // const char* kAttrName = "_output_shapes";
            // for each output node, set shape
            // for (int i = 0; i < node->num_outputs(); ++i){
            // Status s = ic->MakeShapeFromShapeProto(p, &h);
            // if (!s.ok()) {
            // return errors::InvalidArgument("Node '", node->name(), " has an invalid ",
            // kAttrName, " attribute (shape #", i,
            // " error:'", s.error_message(), "'");
            // }
            // s = refiner_->SetShape(node, i, h);
            // if(!s.ok()){
            // return errors::InvalidArgument(
            // "Node '", node->name(), "' has an ", kAttrName,
            // " attribute inconsistent with the GraphDef for output #", i, ": ",
            // s.error_message());
            // }
            // }
            return Status::OK();
        }

        // sort in topological order
        Status GraphConstructor::Convert() {
            std::vector<InputInfo> inputs;
            int processed = 0;
            while(!ready_.empty()){
                // get node from ready_, just process it
                int o = *ready_.begin();
                ready_.erase(ready_.begin());
                ++processed;
                NodeDef node_def = consume_node_def(o);

                // generate inputs
                for (int i = 0; i < node_def.input_size(); ++i){
                    TensorId tensor_id = ParseTensorName(node_def.input(i));
                    Node* src_node;
                    int src_index;
                    // Locate input in newly-imported nodes
                    auto iter = gdef_nodes_.find(tensor_id.node());
                    DCHECK(iter != gdef_nodes_.end()) << tensor_id.node();
                    src_node = iter->second.node;
                    src_index = tensor_id.index();

                    // validate src_index
                    if (src_node != nullptr && src_index >= src_node->num_outputs()) {
                        std::ostringstream out;
                        out << "Node '" << node_def.name() << "': Connecting to invalid output "
                            << tensor_id.index() << " of source node " << tensor_id.node()
                            << " which has " << src_node->num_outputs() << " outputs.";

                        if (src_node->type_string() == "If" ||
                                src_node->type_string() == "StatelessIf" ||
                                src_node->type_string() == "While" ||
                                src_node->type_string() == "StatelessWhile") {
                            out << " Try using "
                                << "tf.compat.v1.experimental.output_all_intermediates(True).";
                        }
                        return errors::InvalidArgument(out.str());
                    }

                    inputs.emplace_back(string(tensor_id.node()), src_node, src_index);
                }

                // populate the remain properties
                const OpDef* op_def;
                TF_RETURN_IF_ERROR(
                        g_->op_registry()->LookUpOpDef(node_def.op(), &op_def));
                if (opts_.add_default_attributes) {
                    AddDefaultsToNodeDef(*op_def, &node_def);
                }
                if (opts_.validate_nodes) {
                    TF_RETURN_IF_ERROR(ValidateNodeDef(node_def, *op_def));
                }
                Node* node;

                // final build node
                TF_RETURN_IF_ERROR(MakeNode(std::move(node_def), &node));
                // assign Node to NodeInfo
                gdef_nodes_[node->name()].node = node;

                // Add edges from inputs to *node to the graph.
                for (size_t i = 0; i < inputs.size(); ++i){
                    TF_RETURN_IF_ERROR(MakeEdge(inputs[i].node, inputs[i].index, node, i));
                }
                // infer shape of node
                TF_RETURN_IF_ERROR(ValidateShape(node));

                // Update pending_count_ for outputs.
                UpdatePendingCountAndReady(o);
            }

            if (processed < node_def_count()) {
                LOG(WARNING) << "IN " << __func__ << " " << (node_def_count() - processed)
                    << " NODES IN A CYCLE";
                for (int64 i = 0; i < node_def_count(); i++) {
                    if (pending_count_[i] != 0) {
                        LOG(WARNING) << "PENDING: " << SummarizeNodeDef(get_node_def(i))
                            << " WITH PENDING COUNT = " << pending_count_[i];
                    }
                }
                // PrintCycles();
                return errors::InvalidArgument(node_def_count() - processed,
                        " nodes in a cycle");
            }

            return Status::OK();
        }

        Status GraphConstructor::MakeNode(NodeDef&& node_def, Node** node){
            // Add the node to the graph.
            Status status;
            *node = g_->AddNode(std::move(node_def), &status);
            if (!status.ok()) return status;
            // (*node)->set_assigned_device_name((*node)->def().device());
            return Status::OK();
        }

        Status GraphConstructor::MakeEdge(Node* src, int output_index, Node* dst,
                int input_index) {
            DataType src_out = src->output_type(output_index);
            DataType dst_in = dst->input_type(input_index);
            // type is the same in two ends in each edge
            if (!TypesCompatible(dst_in, src_out)) {
                return errors::InvalidArgument(
                        "Input ", input_index, " of node ", dst->name(), " was passed ",
                        DataTypeString(src_out), " from ", src->name(), ":", output_index,
                        " incompatible with expected ", DataTypeString(dst_in), ".");
            }
            g_->AddEdge(src, output_index, dst, input_index);
            return Status::OK();
        }

        void GraphConstructor::UpdatePendingCountAndReady(int processed) {
            // update ready_ after process the node(the processed one)
            for (size_t i = 0; i < outputs_[processed].size(); ++i) {
                const int output = outputs_[processed][i];
                int* current_pending_count = &pending_count_[output];
                CHECK_GT(*current_pending_count, 0);
                (*current_pending_count)--;
                if (*current_pending_count == 0) {
                    ready_.insert(output);
                }
            }
        }
        void GraphConstructor::Undo(){}
        void GraphConstructor::FixupSourceAndSinkEdges(){}

        Status GraphConstructor::EnsureNoNameCollisions(){return Status::OK();}
        Status GraphConstructor::PopulateReturnTensors(){return Status::OK();}
        Status GraphConstructor::PopulateReturnNodes(){return Status::OK();}
        Status GraphConstructor::PopulateMissingUnusedInputMapKeys(){return Status::OK();}
    } // namespace


    Status ConvertGraphDefToGraph(const GraphConstructorOptions& opts,
            const GraphDef& gdef, Graph* g) {
        return ConvertGraphDefToGraph(opts, GraphDef(gdef), g);
    }

    Status ConvertGraphDefToGraph(const GraphConstructorOptions& opts,
            GraphDef&& gdef, Graph* g) {
        ShapeRefiner refiner(g->op_registry());
        return GraphConstructor::Construct(opts, std::move(gdef), g, &refiner);
    }

}
