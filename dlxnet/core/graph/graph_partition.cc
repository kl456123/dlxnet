#include "dlxnet/core/graph/graph_partition.h"


namespace dlxnet{

    Status Partition(const PartitionOptions& opts, Graph* input,
            std::unordered_map<string, GraphDef>* partitions){
        // for each node, partition them according their placements
        for(const Node* dst : input->nodes()){
        }
        // used to debug
        string dstp = "/device:CPU:0";
        GraphDef* dst_graph = &(*partitions)[dstp];
        input->ToGraphDef(dst_graph);
        return Status::OK();
    }
}
