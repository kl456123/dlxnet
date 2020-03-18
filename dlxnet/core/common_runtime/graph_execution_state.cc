#include <memory>

#include "dlxnet/core/common_runtime/graph_execution_state.h"
#include "dlxnet/core/graph/graph_constructor.h"
#include "dlxnet/core/platform/macros.h"
#include "dlxnet/core/framework/op.h"



namespace dlxnet{
    /*static*/ Status GraphExecutionState::MakeForBaseGraph(
            GraphDef&& graph_def, const GraphExecutionStateOptions& options,
            std::unique_ptr<GraphExecutionState>* out_state){

        if(options.session_options->config.graph_options().place_pruned_graph()){
            // just save original graph def is ok, no need to place original graph,
            // then just place pruned graph.

        }else{
            auto ret = absl::WrapUnique(new GraphExecutionState(nullptr, options));
            auto base_graph = absl::make_unique<Graph>(OpRegistry::Global());
            TF_RETURN_IF_ERROR(
                    ConvertGraphDefToGraph({}, std::move(graph_def), base_graph.get()));
            TF_RETURN_IF_ERROR(ret->InitBaseGraph(std::move(base_graph)));
            *out_state = std::move(ret);
        }
    }

    /*static*/ Status GraphExecutionState::MakeForPrunedGraph(
            GraphDef&& graph_def, const GraphExecutionStateOptions& options,
            std::unique_ptr<GraphExecutionState>* out_state){
    }

    GraphExecutionState::GraphExecutionState(std::unique_ptr<GraphDef>&& graph_def,
            const GraphExecutionStateOptions& options)
        : original_graph_def_(graph_def),
        device_set_(options.device_set),
        session_options_(options.session_options),
        session_handle_(options.session_handle),
        graph_(nullptr) {}
}
