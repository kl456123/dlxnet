#include "dlxnet/core/graph/subgraph.h"
#include "dlxnet/core/graph/tensor_id.h"

#include <memory>
#include <vector>

namespace dlxnet{
    namespace subgraph{
        namespace {
            typedef std::unordered_map<StringPiece, Node*, StringPieceHasher> NameIndex;

            // Rewrite graph by replacing the output tensors specified in
            // "fed_outputs" with special feed nodes for each specified output
            // tensor, and removing any nodes that are now disconnected from the
            // part of the graph that reaches the sink node.  The set of special
            // feed nodes added to the graph are returned in "*feed_nodes".
            //
            // Return true on success.  On error, return false and sets *error to
            // an appropriate error message (and *g is left in an indeterminate
            // state).
            Status FeedInputs(
                    Graph* g, const std::vector<std::unique_ptr<PruneRewrite>>& feed_rewrites,
                    NameIndex* name_index, DataTypeVector* out_feed_types){
                out_feed_types->clear();
                out_feed_types->reserve(feed_rewrites.size());
                for (size_t i = 0; i < feed_rewrites.size(); ++i) {
                    const string& t = feed_rewrites[i]->endpoint_name();
                    TensorId id(ParseTensorName(t));

                    auto iter = name_index->find(id.first);
                    if (iter == name_index->end()) {
                        return errors::NotFound("FeedInputs: unable to find feed output ", t);
                    }
                    Node* n = iter->second;
                    DCHECK_EQ(n->name(), id.first);
                    if (id.second >= n->num_outputs()) {
                        return errors::InvalidArgument(
                                "FeedInputs: ", t, " should have output index < ", n->num_outputs());
                    }

                    Node* feed_node;
                    TF_RETURN_IF_ERROR(
                            feed_rewrites[i]->AddNode(g, {n, id.second}, &feed_node));
                    // Update name_index
                    (*name_index)[feed_node->name()] = feed_node;

                    // Look through edges coming out of "n" for edges whose src_output() index
                    // matches "output_index".  If found, replace the edges with a connection
                    // from the special feed node.
                    std::vector<const Edge*> to_remove;
                    for (const Edge* e : n->out_edges()) {
                        if (e->src_output() == id.second) {
                            to_remove.emplace_back(e);
                        }
                    }

                    for (const Edge* e : to_remove) {
                        if (e->src_output() == id.second) {
                            g->AddEdge(feed_node, 0, e->dst(), e->dst_input());
                        }
                        g->RemoveEdge(e);
                    }
                    out_feed_types->push_back(BaseType(n->output_type(id.second)));
                }
                return Status::OK();
            }


            Status FetchOutputs(
                    Graph* g, const std::vector<std::unique_ptr<PruneRewrite>>& fetch_rewrites,
                    NameIndex* name_index, std::vector<Node*>* out_fetch_nodes,
                    DataTypeVector* out_fetch_types){
                out_fetch_nodes->clear();
                out_fetch_nodes->reserve(fetch_rewrites.size());
                for (size_t i = 0; i < fetch_rewrites.size(); ++i) {
                    const string& t = fetch_rewrites[i]->endpoint_name();

                    // Parse t into node_name and output_index.
                    TensorId id(ParseTensorName(t));

                    // Find node in graph with that name.
                    auto iter = name_index->find(id.first);
                    if (iter == name_index->end()) {
                        return errors::NotFound("FetchOutputs node ", t, ": not found");
                    }
                    Node* n = iter->second;
                    DCHECK_EQ(n->name(), id.first);
                    VLOG(2) << "Found fetch node for " << t;

                    // Validate output_index
                    if (n->num_outputs() == 0) {
                        return errors::InvalidArgument(
                                "Tried to fetch data for '", t,
                                "', which produces no output.  To run to a node but not fetch any "
                                "data, pass '",
                                t,
                                "' as an argument to the 'target_node_names' argument of the "
                                "Session::Run API.");
                    } else if (id.second >= n->num_outputs()) {
                        return errors::InvalidArgument("FetchOutputs ", t,
                                ": output index too large, must be < ",
                                n->num_outputs());
                    }

                    // Create the fetch Node and connect it up
                    Node* fetch_node;
                    TF_RETURN_IF_ERROR(
                            fetch_rewrites[i]->AddNode(g, {n, id.second}, &fetch_node));
                    // Update the index.
                    (*name_index)[fetch_node->name()] = fetch_node;
                    out_fetch_nodes->push_back(fetch_node);
                    out_fetch_types->push_back(BaseType(n->output_type(id.second)));
                }
                return Status::OK();
            }

            Status PruneForTargets(Graph* g, const NameIndex& name_index,
                    const std::vector<Node*>& fetch_nodes,
                    const gtl::ArraySlice<string>& target_nodes){
                return Status::OK();
            }
        }// namespace

        Status ArgFeedRewrite::AddNode(Graph* g, NodeBuilder::NodeOut feed_tensor,
                Node** out_node) {
            // NOTE(mrry): We must include the index as part of the node
            // name, because _Arg is a "stateful" kernel and therefore
            // its name must uniquely identify a kernel instance across all
            // graphs in the same session.
            TF_RETURN_IF_ERROR(
                    NodeBuilder(strings::StrCat("_arg_", feed_tensor.node->name(), "_",
                            feed_tensor.index, "_", arg_index_),
                        "_Arg")
                    .Attr("T", BaseType(feed_tensor.node->output_type(feed_tensor.index)))
                    .Attr("index", arg_index_)
                    .Finalize(g, out_node));
            return Status::OK();
        }

        Status RetvalFetchRewrite::AddNode(Graph* g, NodeBuilder::NodeOut fetch_tensor,
                Node** out_node) {
            // NOTE(mrry): We must include the index as part of the node
            // name, because _Retval is a "stateful" kernel and therefore
            // its name must uniquely identify a kernel instance across all
            // graphs in the same session.
            TF_RETURN_IF_ERROR(
                    NodeBuilder(strings::StrCat("_retval_", fetch_tensor.node->name(), "_",
                            fetch_tensor.index, "_", retval_index_),
                        "_Retval")
                    .Input(fetch_tensor.node, fetch_tensor.index)
                    .Attr("T",
                        BaseType(fetch_tensor.node->output_type(fetch_tensor.index)))
                    .Attr("index", retval_index_)
                    .Finalize(g, out_node));
            (*out_node)->set_assigned_device_name(device_info().name());
            return Status::OK();
        }




        Status RewriteGraphForExecution(
                Graph* g, const gtl::ArraySlice<string>& fed_outputs,
                const gtl::ArraySlice<string>& fetch_outputs,
                const gtl::ArraySlice<string>& target_node_names,
                const DeviceAttributes& device_info,
                RewriteGraphMetadata* out_metadata){
            std::vector<std::unique_ptr<PruneRewrite>> feed_rewrites;
            feed_rewrites.reserve(fed_outputs.size());
            for (size_t i = 0; i < fed_outputs.size(); ++i) {
                feed_rewrites.emplace_back(new ArgFeedRewrite(
                            &fed_outputs[i], &device_info, static_cast<int32>(i)));
            }


            std::vector<std::unique_ptr<PruneRewrite>> fetch_rewrites;
            fetch_rewrites.reserve(fetch_outputs.size());
            for (size_t i = 0; i < fetch_outputs.size(); ++i) {
                fetch_rewrites.emplace_back(new RetvalFetchRewrite(
                            &fetch_outputs[i], &device_info, static_cast<int32>(i)));
            }

            return RewriteGraphForExecution(g, feed_rewrites, fetch_rewrites,
                    target_node_names, out_metadata);
        }

        // A more general version of the above function that supports
        // customizable rewriting actions for each fed and fetched tensor.
        Status RewriteGraphForExecution(
                Graph* g, const std::vector<std::unique_ptr<PruneRewrite>>& feed_rewrites,
                const std::vector<std::unique_ptr<PruneRewrite>>& fetch_rewrites,
                const gtl::ArraySlice<string>& target_node_names,
                RewriteGraphMetadata* out_metadata){
            //////////////// some insanity checks
            if (fetch_rewrites.empty() && target_node_names.empty()) {
                return errors::InvalidArgument(
                        "Must specify at least one target to fetch or execute.");
            }

            std::unordered_set<string> endpoints;
            for (const auto& feed_rewrite : feed_rewrites) {
                auto result = endpoints.insert(feed_rewrite->endpoint_name());
                if (!result.second) {
                    return errors::InvalidArgument("Endpoint \"",
                            feed_rewrite->endpoint_name(),
                            "\" fed more than once.");
                }
            }

            for (const auto& fetch_rewrite : fetch_rewrites) {
                if (endpoints.count(fetch_rewrite->endpoint_name()) > 0) {
                    return errors::InvalidArgument(fetch_rewrite->endpoint_name(),
                            " is both fed and fetched.");
                }
            }

            // insert feed input and fetch output
            // A separate index mapping name to Node*, for use by FeedInputs,
            // FetchOutputs, and PruneForTargets
            NameIndex name_index;
            name_index.reserve(g->num_nodes());
            for (Node* n : g->nodes()) {
                name_index[n->name()] = n;
            }

            // Add the feeds.  This may replace nodes in the graph, including the nodes
            // currently listed in "fetch_rewrites".  We pass "name_index" so the index is
            // kept up to date.
            if (!feed_rewrites.empty()) {
                TF_RETURN_IF_ERROR(
                        FeedInputs(g, feed_rewrites, &name_index, &out_metadata->feed_types));
            }

            // Add the fetch nodes, also updating "name_index".
            std::vector<Node*> fetch_nodes;
            if (!fetch_rewrites.empty()) {
                TF_RETURN_IF_ERROR(FetchOutputs(g, fetch_rewrites, &name_index,
                            &fetch_nodes, &out_metadata->fetch_types));
            }

            // Prune the graph to only compute what is needed for the fetch nodes and the
            // target nodes.
            if (!fetch_nodes.empty() || !target_node_names.empty()) {
                TF_RETURN_IF_ERROR(
                        PruneForTargets(g, name_index, fetch_nodes, target_node_names));
            }

            return Status::OK();
        }
    }//namespace subgraph
}// namespace dlxnet
