#ifndef DLXNET_CORE_GRAPH_SUBGRAPH_H_
#define DLXNET_CORE_GRAPH_SUBGRAPH_H_
#include <vector>
#include <memory>

#include "dlxnet/core/lib/status.h"
#include "dlxnet/core/graph/graph.h"
#include "dlxnet/core/lib/gtl/array_slice.h"
#include "dlxnet/core/framework/device_attributes.pb.h"

namespace dlxnet{
    // Information about a graph rewritten by `RewriteGraphForExecution()`.
    struct RewriteGraphMetadata {
        // The element type of each tensor fed to this subgraph. The order
        // of types corresponds to the order of tensor names in
        // `fed_outputs` when calling `RewriteGraphForExecution()`.
        DataTypeVector feed_types;
        // The element type of each tensor fetched from this subgraph. The
        // order of types corresponds to the order of tensor names in
        // `fetch_outputs` when calling `RewriteGraphForExecution()`.
        DataTypeVector fetch_types;
    };

    class PruneRewrite{
    };



    Status RewriteGraphForExecution(
            Graph* g, const gtl::ArraySlice<string>& fed_outputs,
            const gtl::ArraySlice<string>& fetch_outputs,
            const gtl::ArraySlice<string>& target_node_names,
            const DeviceAttributes& device_info,
            RewriteGraphMetadata* out_metadata);

    // A more general version of the above function that supports
    // customizable rewriting actions for each fed and fetched tensor.
    Status RewriteGraphForExecution(
            Graph* g, const std::vector<std::unique_ptr<PruneRewrite>>& feed_rewrites,
            const std::vector<std::unique_ptr<PruneRewrite>>& fetch_rewrites,
            const gtl::ArraySlice<string>& target_node_names,
            RewriteGraphMetadata* out_metadata);
}


#endif
