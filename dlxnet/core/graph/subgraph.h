#ifndef DLXNET_CORE_GRAPH_SUBGRAPH_H_
#define DLXNET_CORE_GRAPH_SUBGRAPH_H_
#include <vector>
#include <memory>

#include "dlxnet/core/lib/status.h"
#include "dlxnet/core/graph/graph.h"
#include "dlxnet/core/lib/gtl/array_slice.h"
#include "dlxnet/core/framework/device_attributes.pb.h"
#include "dlxnet/core/graph/node_builder.h"

namespace dlxnet{
    namespace subgraph{
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

        // Describes the action to take on a particular tensor endpoint (described by
        // a "<node_name>:<output_index>" pair) when pruning the graph.
        //
        // The `AddNode()` method must be overridden to describe this action. The method
        // will be invoked once during `RewriteGraphForExecution()` with tensor endpoint
        // named by `endpoint_name`, and it may either create a single new node, or fail
        // with an error if the resulting graph would be invalid.
        class PruneRewrite{
            public:
                // `endpoint_name` and `device_info` must outlive this object.
                PruneRewrite(const string* endpoint_name, const DeviceAttributes* device_info)
                    : endpoint_name_(endpoint_name), device_info_(device_info) {}
                virtual ~PruneRewrite() {}

                // Creates a new node whose output replaces the given `tensor` in graph `g`.
                // The node will be assigned to the device named in `device_info`.
                virtual Status AddNode(Graph* g, NodeBuilder::NodeOut tensor,
                        Node** out_node) = 0;

                // Returns the name of the tensor to which this rewrite applies.
                const string& endpoint_name() { return *endpoint_name_; }

            protected:
                // The device on which the new node will be created.
                const DeviceAttributes& device_info() { return *device_info_; }

            private:
                const string* const endpoint_name_;          // Not owned.
                const DeviceAttributes* const device_info_;  // Not owned.
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

        // A rewrite action that adds an _Arg node for a fed tensor.
        class ArgFeedRewrite : public PruneRewrite {
            public:
                ArgFeedRewrite(const string* endpoint_name,
                        const DeviceAttributes* device_info, int32 arg_index)
                    : PruneRewrite(endpoint_name, device_info), arg_index_(arg_index) {}
                Status AddNode(Graph* g, NodeBuilder::NodeOut feed_tensor,
                        Node** out_node) override;

            private:
                const int32 arg_index_;
        };

        // A rewrite action that adds a _Retval node for a fetched tensor.
        class RetvalFetchRewrite : public PruneRewrite {
            public:
                RetvalFetchRewrite(const string* endpoint_name,
                        const DeviceAttributes* device_info, int32 retval_index)
                    : PruneRewrite(endpoint_name, device_info), retval_index_(retval_index) {}
                Status AddNode(Graph* g, NodeBuilder::NodeOut fetch_tensor,
                        Node** out_node) override;

            private:
                const int32 retval_index_;
        };
    }// namespace subgraph
}// namespace dlxnet


#endif
