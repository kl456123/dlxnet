#include "dlxnet/core/graph/graph_constructor.h"
#include "dlxnet/core/platform/macros.h"



namespace dlxnet{

    class GraphConstructor{
        public:
            GraphConstructor(const GraphConstructorOptions& opts,
                    GraphDef&& gdef, Graph* g, ShapeRefiner* refiner)
                :graph_def_(gdef){
                }

            static Status Construct(const GraphConstructorOptions& opts,
                    GraphDef&& gdef, Graph* g, ShapeRefiner* refiner);
            Status TryImport() {
                TF_RETURN_IF_ERROR(EnsureNoNameCollisions());
                TF_RETURN_IF_ERROR(ValidateInputMapAndControlDependencies());
                TF_RETURN_IF_ERROR(BuildNodeIndex());
                TF_RETURN_IF_ERROR(InitFromEdges());

                // NOTE: Convert() invokes `consume_node_def()` on each node in the input
                // graph, so `get_node_def()` is no longer usable once it is called.
                TF_RETURN_IF_ERROR(Convert());

                TF_RETURN_IF_ERROR(AddBackEdges());
                TF_RETURN_IF_ERROR(UpdateVersionDef());
                TF_RETURN_IF_ERROR(PopulateReturnTensors());
                TF_RETURN_IF_ERROR(PopulateReturnNodes());
                TF_RETURN_IF_ERROR(PopulateMissingUnusedInputMapKeys());
                UpdateUniquifiedColocationNames();
                FixupSourceAndSinkEdges(g_);
                return Status::OK();
            }
            void Undo();
        private:
            Status EnsureNoNameCollisions();
            Status ValidateInputMapAndControlDependencies();
            Status BuildNodeIndex();
            Status InitFromEdges();
            Status Convert();
            Status AddBackEdges();
            Status UpdateVersionDef();
            Status PopulateReturnTensors();
            Status PopulateReturnNodes();
            Status PopulateMissingUnusedInputMapKeys();

            // Index of NodeDefs in node_defs_ with all inputs already converted. We use a
            // (sorted) set so nodes are created in the order defined in the GraphDef.
            std::set<int> ready_;
    };


    Status GraphConstructor::Construct(const GraphConstructorOptions& opts,
            GraphDef&& gdef, Graph* g, ShapeRefiner* refiner){
        GraphConstructor c(opts, std::move(graph_def), g, refiner);
        const Status s = c.TryImport();
        if (!s.ok()) c.Undo();
        return s;
    }
    Status BuildNodeIndex(){
        for(int i=0;i<node_def_count();i++){
        }
    }
    Status GraphConstructor::InitFromEdges() {}

    // sort in topological order
    Status GraphConstructor::Convert() {
        int processed = 0;
        while(!ready_.empty()){
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
            PrintCycles();
            return errors::InvalidArgument(node_def_count() - processed,
                    " nodes in a cycle");
        }

        return Status::OK();
    }

    void GraphConstructor::UpdatePendingCountAndReady(int processed,
            bool is_next_iteration) {
        for (size_t i = 0; i < outputs_[processed].size(); ++i) {
            const int output = outputs_[processed][i];
            // We didn't consider NextIteration->Merge edges when computing
            // pending_counts_ so we should not have to consider it here either.
            bool is_next_iteration_to_merge_edge =
                is_next_iteration && merge_node_indices_.count(output) == 1;
            if (!is_next_iteration_to_merge_edge) {
                int* current_pending_count = &pending_count_[output];
                CHECK_GT(*current_pending_count, 0);
                (*current_pending_count)--;
                if (*current_pending_count == 0) {
                    ready_.insert(output);
                }
            }
        }
    }


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
