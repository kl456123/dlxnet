#include <memory>

#include "dlxnet/core/common_runtime/graph_execution_state.h"
#include "dlxnet/core/common_runtime/placer.h"
#include "dlxnet/core/common_runtime/optimization_registry.h"
#include "dlxnet/core/graph/graph_constructor.h"
#include "dlxnet/core/graph/tensor_id.h"
#include "dlxnet/core/platform/macros.h"
#include "dlxnet/core/framework/op.h"



namespace dlxnet{
    namespace {
        template <class Map>
            Status LookupDevice(const DeviceSet& device_set, const string& tensor_name,
                    const Map& tensor2device,
                    const dlxnet::DeviceAttributes** out_device_attrs) {
                *out_device_attrs = nullptr;
                if (tensor2device.empty()) {
                    *out_device_attrs = &device_set.client_device()->attributes();
                    return Status::OK();
                }
                const auto it = tensor2device.find(tensor_name);
                if (it == tensor2device.end()) {
                    *out_device_attrs = &device_set.client_device()->attributes();
                    return Status::OK();
                }
                DeviceNameUtils::ParsedName parsed_name;
                if (!DeviceNameUtils::ParseFullName(it->second, &parsed_name)) {
                    return errors::InvalidArgument("Invalid device name ('", it->second,
                            "') provided for the tensor '", tensor_name,
                            "' in CallableOptions");
                }
                Device* device = device_set.FindDeviceByName(
                        DeviceNameUtils::ParsedNameToString(parsed_name));
                if (device == nullptr) {
                    return errors::InvalidArgument("Device '", it->second,
                            "' specified for tensor '", tensor_name,
                            "' in CallableOptions does not exist");
                }
                *out_device_attrs = &device->attributes();
                return Status::OK();
            }
    }//namespace
    /*static*/ Status GraphExecutionState::MakeForBaseGraph(
            GraphDef&& graph_def, const GraphExecutionStateOptions& options,
            std::unique_ptr<GraphExecutionState>* out_state){

        if(options.session_options->config.graph_options().place_pruned_graph()){
            // just save original graph def is ok, no need to place original graph,
            // then just place pruned graph.

        }else{
            auto ret = absl::WrapUnique(new GraphExecutionState(nullptr, options));
            auto base_graph = absl::make_unique<Graph>(OpRegistry::Global());
            TF_RETURN_IF_ERROR(
                    ConvertGraphDefToGraph({}, std::move(graph_def), base_graph.get()));
            TF_RETURN_IF_ERROR(ret->InitBaseGraph(std::move(base_graph)));
            *out_state = std::move(ret);
        }
    }

    /*static*/ Status GraphExecutionState::MakeForPrunedGraph(
            GraphDef&& graph_def, const GraphExecutionStateOptions& options,
            std::unique_ptr<GraphExecutionState>* out_state){
    }

    GraphExecutionState::GraphExecutionState(std::unique_ptr<GraphDef>&& graph_def,
            const GraphExecutionStateOptions& options)
        : original_graph_def_(std::move(graph_def)),
        device_set_(options.device_set),
        session_options_(options.session_options),
        session_handle_(options.session_handle),
        graph_(nullptr) {}

    Status GraphExecutionState::InitBaseGraph(std::unique_ptr<Graph>&& new_graph){
        // pre optimize
        GraphOptimizationPassOptions optimization_options;
        optimization_options.session_handle = session_handle_;
        optimization_options.session_options = session_options_;
        optimization_options.graph = &new_graph;
        optimization_options.device_set = device_set_;

        TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
                    OptimizationPassRegistry::PRE_PLACEMENT, optimization_options));
        // placement
        Placer placer(new_graph.get(), "", nullptr, device_set_,
                /* default_local_device= */ nullptr,
                session_options_ == nullptr ||
                session_options_->config.allow_soft_placement(),
                session_options_ != nullptr &&
                session_options_->config.log_device_placement());
        // TODO(mrry): Consider making the Placer cancelable.
        TF_RETURN_IF_ERROR(placer.Run());
        // post optimize
        TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
                    OptimizationPassRegistry::POST_PLACEMENT, optimization_options));

        graph_ = new_graph.release();
        return Status::OK();
    }

    Status GraphExecutionState::BuildGraph(const BuildGraphOptions& options,
            std::unique_ptr<ClientGraph>* out){

        VLOG(1) << "BuildGraph";
        const uint64 start_time_usecs = Env::Default()->NowMicros();
        if (!graph_) {
            // It is only valid to call this method directly when the original graph
            // was created with the option `place_pruned_graph == false`.
            return errors::Internal(
                    "Attempted to prune a graph that has not been fully initialized.");
        }

        // Grappler optimization might change the structure of a graph itself, and
        // also it can add/prune functions to/from the library.
        std::unique_ptr<Graph> optimized_graph;
        Status s = OptimizeGraph(options, &optimized_graph);
        if(!s.ok()){
            VLOG(2) << "Grappler optimization failed. Error: " << s.error_message();
            // Simply copy the original graph and the function library if we couldn't
            // optimize it.
            optimized_graph.reset(new Graph(OpRegistry::Global()));
            CopyGraph(*graph_, optimized_graph.get());
        }

        subgraph::RewriteGraphMetadata rewrite_metadata;
        // prune graph
        TF_RETURN_IF_ERROR(
                PruneGraph(options, optimized_graph.get(), &rewrite_metadata));

        // TODO(andydavis): Clarify optimization pass requirements around CostModel.
        GraphOptimizationPassOptions optimization_options;
        optimization_options.session_options = session_options_;
        optimization_options.graph = &optimized_graph;
        optimization_options.device_set = device_set_;

        TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
                    OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, optimization_options));

        std::unique_ptr<ClientGraph> client_graph(new ClientGraph(
                    rewrite_metadata.feed_types, rewrite_metadata.fetch_types));
        CopyGraph(*optimized_graph, &client_graph->graph);
        *out = std::move(client_graph);
        return Status::OK();
    }

    Status GraphExecutionState::PruneGraph(
            const BuildGraphOptions& options, Graph* graph,
            subgraph::RewriteGraphMetadata* out_rewrite_metadata) {
        std::vector<std::unique_ptr<subgraph::PruneRewrite>> feed_rewrites;
        feed_rewrites.reserve(options.callable_options.feed_size());
        std::vector<std::unique_ptr<subgraph::PruneRewrite>> fetch_rewrites;
        fetch_rewrites.reserve(options.callable_options.fetch_size());
        // feed nodes
        for (int i = 0; i < options.callable_options.feed_size(); ++i) {
            // WARNING: feed MUST be a reference, since ArgFeedRewrite and
            // tensors_and_devices holds on to its address.
            const string& feed = options.callable_options.feed(i);
            const DeviceAttributes* device_info;
            TF_RETURN_IF_ERROR(LookupDevice(*device_set_, feed,
                        options.callable_options.feed_devices(),
                        &device_info));
            feed_rewrites.emplace_back(
                    new subgraph::ArgFeedRewrite(&feed, device_info, i));
        }

        if (!options.callable_options.fetch_devices().empty() &&
                !options.callable_options.fetch_skip_sync()) {
            return errors::Unimplemented(
                    "CallableOptions.fetch_skip_sync = false is not yet implemented. You "
                    "can set it to true instead, but MUST ensure that Device::Sync() is "
                    "invoked on the Device corresponding to the fetched tensor before "
                    "dereferencing the Tensor's memory.");
        }

        // fetch nodes
        for (int i = 0; i < options.callable_options.fetch_size(); ++i) {
            // WARNING: fetch MUST be a reference, since RetvalFetchRewrite and
            // tensors_and_devices holds on to its address.
            const string& fetch = options.callable_options.fetch(i);
            const DeviceAttributes* device_info;
            TF_RETURN_IF_ERROR(LookupDevice(*device_set_, fetch,
                        options.callable_options.fetch_devices(),
                        &device_info));
            fetch_rewrites.emplace_back(
                    new subgraph::RetvalFetchRewrite(&fetch, device_info, i));
        }

        // target nodes
        std::vector<string> target_node_names(
                options.callable_options.target().begin(),
                options.callable_options.target().end());

        // rewrite graph
        TF_RETURN_IF_ERROR(subgraph::RewriteGraphForExecution(
                    graph, feed_rewrites, fetch_rewrites, target_node_names,
                    out_rewrite_metadata));
        return Status::OK();
    }

    Status GraphExecutionState::OptimizeGraph(
            const BuildGraphOptions& options, std::unique_ptr<Graph>* optimized_graph) {
        if (session_options_->config.graph_options().place_pruned_graph()) {
            return errors::InvalidArgument("Can't optimize a pruned graph");
        }
        // use grappler
        return errors::InvalidArgument("Meta Optimizer disabled");
    }
}
