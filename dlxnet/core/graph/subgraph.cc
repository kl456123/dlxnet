#include "dlxnet/core/graph/subgraph.h"

namespace dlxnet{

    Status RewriteGraphForExecution(
            Graph* g, const gtl::ArraySlice<string>& fed_outputs,
            const gtl::ArraySlice<string>& fetch_outputs,
            const gtl::ArraySlice<string>& target_node_names,
            const DeviceAttributes& device_info,
            RewriteGraphMetadata* out_metadata){}

    // A more general version of the above function that supports
    // customizable rewriting actions for each fed and fetched tensor.
    Status RewriteGraphForExecution(
            Graph* g, const std::vector<std::unique_ptr<PruneRewrite>>& feed_rewrites,
            const std::vector<std::unique_ptr<PruneRewrite>>& fetch_rewrites,
            const gtl::ArraySlice<string>& target_node_names,
            RewriteGraphMetadata* out_metadata){}
}// namespace dlxnet
