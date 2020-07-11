#include "absl/container/flat_hash_map.h"

#include "dlxnet/core/framework/node_def_util.h"
#include "dlxnet/core/framework/memory_types.h"
#include "dlxnet/core/graph/graph_partition.h"
#include "dlxnet/core/util/device_name_utils.h"

namespace dlxnet{
    namespace{
        // Add an input to dst that comes from the "src_slot" output of the
        // node named by "src_name".
        void AddInput(NodeDef* dst, StringPiece src_name, int src_slot) {
            if (src_slot == 0) {
                dst->add_input(src_name.data(), src_name.size());
            } else {
                dst->add_input(strings::StrCat(src_name, ":", src_slot));
            }
        }

        // A map used to store memory types for the inputs/outputs of every node.
        // The key is a pair of ints consisting of a node id and input/output index.
        // TODO(power): migrate back to std::pair when absl::Hash is fixed for MSVC.
        struct NodePort {
            int node_id;
            int index;

            friend bool operator==(const NodePort& x, const NodePort& y) {
                return x.node_id == y.node_id && x.index == y.index;
            }

            template <typename H>
                friend H AbslHashValue(H h, const NodePort& c) {
                    return H::combine(std::move(h), c.node_id, c.index);
                }
        };

        typedef absl::flat_hash_map<NodePort, MemoryType> MemoryTypeMap;

        // We collect the following information about the graph before performing
        // graph partitioning.
        struct GraphInfo {
            std::vector<DeviceType> device_types;
            MemoryTypeMap input_types;
            MemoryTypeMap output_types;
        };

        DataType EdgeType(const Edge* e) {
            return e->dst()->input_type(e->dst_input());
        }

        // Return true iff we need to add the same device send/recv for 'edge'.
        bool NeedSameDeviceSendRecv(const Edge* edge, const GraphInfo& info) {
            // if (edge->IsControlEdge()) {
            // return false;
            // }
            const Node* src = edge->src();
            const Node* dst = edge->dst();
            if (src->assigned_device_name() == dst->assigned_device_name()) {
                int src_port = edge->src_output();
                int dst_port = edge->dst_input();
                if (info.device_types[src->id()] != DEVICE_CPU) {
                    auto src_it = info.output_types.find({src->id(), src_port});
                    DCHECK(src_it != info.output_types.end());
                    auto dst_it = info.input_types.find({dst->id(), dst_port});
                    DCHECK(dst_it != info.input_types.end());
                    return src_it->second != dst_it->second;
                }
            }
            return false;
        }

        // If 'ndef' is a Send or Recv, fills its attr send_device_incarnation
        // if possible.
        void SetIncarnation(const PartitionOptions& opts, NodeDef* ndef) {
            StringPiece op(ndef->op());
            if (op != "_Send" && op != "_Recv") {
                // Not related to send/recv.
                return;
            }
            const string& send_device = GetNodeAttrString(*ndef, "send_device");
            if (send_device.empty()) {
                // No known send_device. The runtime will detect it later.
                return;
            }
            int64 incarnation = PartitionOptions::kIllegalIncarnation;
            if (!TryGetNodeAttr(*ndef, "send_device_incarnation", &incarnation) ||
                    (incarnation == PartitionOptions::kIllegalIncarnation)) {
                incarnation = opts.get_incarnation(send_device);
                SetAttrValue(incarnation,
                        &((*ndef->mutable_attr())["send_device_incarnation"]));
            }
        }

        // Sets attribute send_device_incarnation of all Send/Recv nodes in
        // 'gdef', if possible.
        void SetIncarnation(const PartitionOptions& opts, GraphDef* gdef) {
            for (NodeDef& ndef : *gdef->mutable_node()) {
                SetIncarnation(opts, &ndef);
            }
            // for (FunctionDef& fdef : *gdef->mutable_library()->mutable_function()) {
            // for (NodeDef& ndef : *fdef.mutable_node_def()) {
            // SetIncarnation(opts, &ndef);
            // }
            // }
        }

        // Build memory and device type info for every node in the graph.
        // TODO(yuanbyu): It might be simpler if we convert MemoryType to
        // DeviceType for the inputs/outputs of each node.
        Status BuildMemoryDeviceInfo(const Graph& g, GraphInfo* info) {
            MemoryTypeVector input_memory_types;
            MemoryTypeVector output_memory_types;

            info->device_types.resize(g.num_node_ids(), DEVICE_CPU);
            for (const Node* node : g.op_nodes()) {
                DeviceNameUtils::ParsedName parsed;
                if (!DeviceNameUtils::ParseFullName(node->assigned_device_name(),
                            &parsed)) {
                    return errors::Internal("Malformed assigned device '",
                            node->assigned_device_name(), "'");
                }

                TF_RETURN_IF_ERROR(MemoryTypesForNode(
                            g.op_registry(), DeviceType(parsed.type), node->def(),
                            &input_memory_types, &output_memory_types));

                int node_id = node->id();
                info->device_types[node_id] = DeviceType(parsed.type);
                for (int i = 0; i < input_memory_types.size(); ++i) {
                    info->input_types[{node_id, i}] = input_memory_types[i];
                }
                for (int i = 0; i < output_memory_types.size(); ++i) {
                    info->output_types[{node_id, i}] = output_memory_types[i];
                }
            }
            return Status::OK();
        }
    } // namespace

    Status Partition(const PartitionOptions& opts, Graph* g,
            std::unordered_map<string, GraphDef>* partitions){
        Status status;
        partitions->clear();

        GraphInfo g_info;

        // At this point, all the graph mutations have been done. Build memory
        // and device type info for every node and edge in the graph.
        status = BuildMemoryDeviceInfo(*g, &g_info);
        if (!status.ok()) return status;

        string dstp;
        std::vector<const Edge*> inputs;

        // for each node, partition them according their placements
        for(const Node* dst : g->nodes()){
            dstp = opts.node_to_loc(dst);
            GraphDef* dst_graph = &(*partitions)[dstp];
            NodeDef* dst_def = dst_graph->add_node();
            *dst_def = dst->def();
            dst_def->set_device(dst->assigned_device_name());
            dst_def->clear_input();  // Inputs are filled below

            // Arrange the incoming edges to dst so that input[i] holds the
            // input flowing into slot numbered i. Trailing entries in input[]
            // hold control edges.
            inputs.clear();
            inputs.resize(dst->num_inputs(), nullptr);
            int32 num_input_edges = 0;
            for (const Edge* edge : dst->in_edges()) {
                DCHECK(inputs[edge->dst_input()] == nullptr);
                inputs[edge->dst_input()] = edge;
                ++num_input_edges;
            }

            if (num_input_edges != dst->num_inputs()) {
                return errors::InvalidArgument("Incomplete graph, missing ",
                        (dst->num_inputs() - num_input_edges),
                        " inputs for ", dst->name());
            }

            // Process in order so that all data edges are added as inputs to
            // dst in Edge::dst_input() order.
            for (const Edge* edge : inputs) {
                const Node* src = edge->src();
                if (!src->IsOp()) continue;  // Skip Sink/Source nodes.

                GraphDef* src_graph = &(*partitions)[opts.node_to_loc(src)];
                if (src_graph == dst_graph&&!NeedSameDeviceSendRecv(edge, g_info)) {
                    // Same partition and compatible memory types:
                    AddInput(dst_def, src->name(), edge->src_output());
                    continue;
                }
            }
        }

        // Set versions, function library and send/recv incarnation.
        for (auto& it : *partitions) {
            GraphDef* gdef = &it.second;
            // *gdef->mutable_versions() = g->versions();
            // Prune unreachable functions from `flib_def` before adding them to `gdef`.
            // *gdef->mutable_library() = flib_def->ReachableDefinitions(*gdef).ToProto();

            // Traverse the graph to fill every send/recv op's incarnation
            // information.
            SetIncarnation(opts, gdef);
        }

        // VLOG(1) << "Added send/recv: controls=" << num_control
            // << ", data=" << num_data;
        if (VLOG_IS_ON(2)) {
            for (auto& it : *partitions) {
                GraphDef* gdef = &it.second;
                // DumpGraphDefToFile(strings::StrCat("partition_", it.first, "_",
                // reinterpret_cast<uintptr_t>(gdef)),
                // *gdef);
            }
        }
        // used to debug
        // string dstp = "/device:CPU:0";
        // GraphDef* dst_graph = &(*partitions)[dstp];
        // input->ToGraphDef(dst_graph);
        return Status::OK();
    }
}
