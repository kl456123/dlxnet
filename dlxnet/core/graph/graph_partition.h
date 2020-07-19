#ifndef DLXNET_CORE_GRAPH_GRAPH_PARTITION_H_
#define DLXNET_CORE_GRAPH_GRAPH_PARTITION_H_
#include <unordered_map>
#include <functional>

#include "dlxnet/core/framework/graph.pb.h"
#include "dlxnet/core/framework/types.pb.h"
#include "dlxnet/core/graph/graph.h"
#include "dlxnet/core/lib/core/status.h"

namespace dlxnet{
    struct PartitionOptions{
        // A function that returns a location for the execution of a given
        // Node.
        typedef std::function<string(const Node*)> NodeToLocFunc;
        NodeToLocFunc node_to_loc = nullptr;

        // A function that returns a unique graph node name with the given
        // prefix.
        typedef std::function<string(const string&)> NewNameFunc;
        NewNameFunc new_name = nullptr;

        // A function that returns the incarnation of a device given the
        // device's fullname. If not found, GetIncarnationFunc should return
        // kIllegalIncarnation.
        static const uint64 kIllegalIncarnation = 0;
        typedef std::function<uint64(const string&)> GetIncarnationFunc;
        GetIncarnationFunc get_incarnation = nullptr;

        // A function that returns the data type into which the tensor
        // should be cast before sent over the wire.
        typedef std::function<DataType(const Edge*)> ShouldCastFunc;
        ShouldCastFunc should_cast = nullptr;
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
