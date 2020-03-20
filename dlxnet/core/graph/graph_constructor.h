#ifndef DLXNET_CORE_GRAPH_GRAPH_CONSTRUCTOR_H_
#define DLXNET_CORE_GRAPH_GRAPH_CONSTRUCTOR_H_
#include "dlxnet/core/graph/graph.h"
#include "dlxnet/core/framework/graph.pb.h"

namespace dlxnet{
    // options for construct graph
    struct GraphConstructorOptions{
        // If true, allows internal ops in the GraphDef.
        bool allow_internal_ops = false;

        // If true, the graph def is expected to have fully specified
        // devices for all nodes. A node in the resulting graph "g" has the
        // device name set accordingly.
        //
        // TODO(zhifengc): if possible, consider removing this option.
        bool expect_device_spec = false;

        // If true, validates that nodes being converted have all expected attrs
        // set and no unknonw attrs set by calling ValidateNodeDef().
        // Setting validate_nodes without add_default_attributes, will fail if
        // the GraphDef does not have all required attributes set.
        bool validate_nodes = false;

        // If true, GraphConstructor will add attributes with their default
        // value to the Node when they are missing from the NodeDef.
        bool add_default_attributes = true;

        bool validate_shape = true;
    };
    // construct from empty graph(sink and source node)
    Status ConvertGraphDefToGraph(const GraphConstructorOptions& opts,
            const GraphDef& gdef, Graph* g);
    Status ConvertGraphDefToGraph(const GraphConstructorOptions& opts,
            GraphDef&& gdef, Graph* g);

}


#endif
