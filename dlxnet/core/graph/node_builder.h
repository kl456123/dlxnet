#ifndef DLXNET_CORE_GRAPH_NODE_BUILDER_H_
#define DLXNET_CORE_GRAPH_NODE_BUILDER_H_
#include "dlxnet/core/platform/types.h"
#include "dlxnet/core/lib/status.h"

namespace dlxnet{
    // helper class to build node(add node to graph)
    // It using NodeDefBuilder internally
    class NodeBuilder{
        public:
            NodeBuilder& Device(const string& string);
            Status Finalize();
    };
}


#endif
