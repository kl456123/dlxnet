#ifndef DLXNET_CORE_GRAPH_GRAPH_PARTITION_H_
#define DLXNET_CORE_GRAPH_GRAPH_PARTITION_H_
#include <unordered_map>

#include "dlxnet/core/framework/graph.pb.h"
#include "dlxnet/core/graph/graph.h"
#include "dlxnet/core/lib/core/status.h"

namespace dlxnet{
    struct PartitionOptions{
        // A function that returns a location for the execution of a given
        // Node.
        typedef std::function<string(const Node*)> NodeToLocFunc;
        NodeToLocFunc node_to_loc = nullptr;
    };
    // Partition "input" graph into a set of graphs, one per location.
    // The location for node n is derived by calling opts.node_to_loc(n).
    // New nodes added by Partition use "opts.new_name(old_name)" to
    // generate node names.
    //
    // Stores the partitions in *partitions.
    Status Partition(const PartitionOptions& opts, Graph* input,
            std::unordered_map<string, GraphDef>* partitions);
}


#endif
