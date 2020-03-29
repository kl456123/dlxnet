#ifndef DLXNET_CORE_COMMON_RUNTIME_GRAPH_EXECUTION_STATE_H_
#define DLXNET_CORE_COMMON_RUNTIME_GRAPH_EXECUTION_STATE_H_
#include <memory>
#include "dlxnet/core/graph/graph.h"
#include "dlxnet/core/platform/macros.h"
#include "dlxnet/core/common_runtime/device_set.h"
#include "dlxnet/core/lib/status.h"
#include "dlxnet/core/framework/graph.pb.h"
#include "dlxnet/core/public/session_options.h"
#include "dlxnet/core/common_runtime/build_graph_options.h"



// all things about graph, including original graph, subgraph and its processed graph
//
//
// processes
// 1. convert graph_def to graph
// 2. optimization pass
// 3. graph placement


namespace dlxnet{
    // options(param for constructor) for GraphExecutionState
    struct GraphExecutionStateOptions{
        const DeviceSet* device_set = nullptr;
        const SessionOptions* session_options = nullptr;
        // Unique session identifier. Can be empty.
        string session_handle;
    };

    class GraphExecutionState{
        public:
            // base graph
            static Status MakeForBaseGraph(
                    GraphDef&& graph_def, const GraphExecutionStateOptions& options,
                    std::unique_ptr<GraphExecutionState>* out_state);
            // pruned graph
            static Status MakeForPrunedGraph(
                    GraphDef&& graph_def, const GraphExecutionStateOptions& options,
                    std::unique_ptr<GraphExecutionState>* out_state);

            // Builds a ClientGraph (a sub-graph of the full graph as induced by
            // the Node set specified in "options").  If successful, returns OK
            // and the caller takes the ownership of "*out". Otherwise, returns
            // an error.
            Status BuildGraph(const BuildGraphOptions& options,
                    std::unique_ptr<Graph>* out);

            // The graph returned by BuildGraph may contain only the pruned
            // graph, whereas some clients may want access to the full graph.
            const Graph* full_graph() { return graph_; }

        private:
            GraphExecutionState(std::unique_ptr<GraphDef>&& graph_def,
                    const GraphExecutionStateOptions& options);

            // Extract the subset of the graph that needs to be run, adding feed/fetch
            // ops as needed.
            Status PruneGraph(const BuildGraphOptions& options, Graph* graph);

            Status OptimizeGraph(
                    const BuildGraphOptions& options, std::unique_ptr<Graph>* optimized_graph);

            Status InitBaseGraph(std::unique_ptr<Graph>&& graph);

            const std::unique_ptr<GraphDef> original_graph_def_;

            // The dataflow graph owned by this object.
            Graph* graph_;

            // Unique session identifier. Can be empty.
            string session_handle_;

            const DeviceSet* device_set_;            // Not owned
            const SessionOptions* session_options_;  // Not owned

            TF_DISALLOW_COPY_AND_ASSIGN(GraphExecutionState);
    };
}


#endif
