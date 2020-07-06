#ifndef DLXNET_CORE_GRAPH_DEFAULT_DEVICE_H_
#define DLXNET_CORE_GRAPH_DEFAULT_DEVICE_H_

#include <string>

#include "dlxnet/core/framework/graph.pb.h"
#include "dlxnet/core/framework/node_def.pb.h"

namespace dlxnet {
    namespace graph {

        // Sets the default device for all nodes in graph_def to "device",
        // only if not already set.
        inline void SetDefaultDevice(const string& device, GraphDef* graph_def) {
            for (int i = 0; i < graph_def->node_size(); ++i) {
                auto node = graph_def->mutable_node(i);
                if (node->device().empty()) {
                    node->set_device(device);
                }
            }
        }

    }  // namespace graph
}  // namespace dlxnet

#endif  // DLXNET_CORE_GRAPH_DEFAULT_DEVICE_H_
