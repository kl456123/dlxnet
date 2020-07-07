#include "dlxnet/core/common_runtime/placer.h"
#include "dlxnet/core/platform/logging.h"
#include "dlxnet/core/common_runtime/colocation_graph.h"


namespace dlxnet{
    namespace{
        // Returns true if the node has no inputs and produces outputs
        // that are consumed by a single node.
        //
        // TODO(vrv): Currently this handles only nodes with one output, but
        // this could be extended to handle the case where a node has many
        // outputs that are connected to nodes in the same colocation group.
        bool IsGeneratorNode(const Node* node) {
            return node->num_inputs() == 0 && node->num_outputs() == 1 &&
                !IsRefType(node->output_type(0));
        }

        void LogDeviceAssignment(const Node* node, bool log_device_placement) {
            // Log placement if log_device_placement is set.
            if (log_device_placement) {
                printf("%s: (%s): %s\n", node->name().c_str(), node->type_string().c_str(),
                        node->assigned_device_name().c_str());
                LOG(INFO) << node->name() << ": "
                    << "(" << node->type_string()
                    << "): " << node->assigned_device_name();
            }
        }

        Status AssignAndLog(int assigned_device, Node* node,
                ColocationGraph* colocation_graph,
                bool log_device_placement) {
            node->set_assigned_device_name_index(assigned_device);

            // Constraint the group of node to the assigned device.
            TF_RETURN_IF_ERROR(colocation_graph->LimitToAssignedDevice(*node));

            LogDeviceAssignment(node, log_device_placement);
            return Status::OK();
        }
    } // namespace
    Placer::Placer(Graph* graph, const string& function_name,
            const FunctionLibraryDefinition* flib_def,
            const DeviceSet* devices, const Device* default_local_device,
            bool allow_soft_placement, bool log_device_placement)
        : graph_(graph),
        function_name_(function_name),
        flib_def_(flib_def),
        devices_(devices),
        default_local_device_(default_local_device),
        allow_soft_placement_(allow_soft_placement),
        log_device_placement_(log_device_placement) {}

    Placer::Placer(Graph* graph, const string& function_name,
            const DeviceSet* devices, const Device* default_local_device)
        : Placer(graph, function_name, nullptr, devices,
                default_local_device, true, false) {}

    Placer::Placer(Graph* graph, const string& function_name,
            const DeviceSet* devices)
        : Placer(graph, function_name, nullptr, devices, nullptr, true,
                false) {}

    Placer::~Placer() {}

    Status Placer::Run() {
        if (devices_->devices().empty()) {
            return errors::FailedPrecondition("No devices are registered");
        }

        if (VLOG_IS_ON(3)) {
            // DumpGraphToFile("placer_input", *graph_, nullptr);
        }
        if (VLOG_IS_ON(5)) {
            for (const Node* node : graph_->op_nodes()) {
                VLOG(5) << "    " << node->name() << ": requested: '"
                    << node->requested_device() << "' assigned: '"
                    << node->assigned_device_name() << "'";
            }
        }

        FunctionStack stack(function_name_);
        ColocationGraph colocation_graph(graph_, stack, flib_def_, devices_,
                default_local_device_, allow_soft_placement_,
                log_device_placement_);

        TF_RETURN_IF_ERROR(colocation_graph.Initialize());

        // For each node, assign a device based on the constraints in the disjoint
        // node set.
        std::vector<Node*> second_pass;
        for (Node* node : graph_->op_nodes()) {
            // The graph may have come pre-populated by the framework with assigned
            // devices (e.g., for stateful placements), so the placer should not try to
            // place nodes that are already placed.
            if (node->has_assigned_device_name()) {
                TF_RETURN_IF_ERROR(colocation_graph.LimitToAssignedDevice(*node));
                LogDeviceAssignment(node, log_device_placement_);
                continue;
            }

            // Heuristic A: prefer to place "generators" with their only
            // consumers.
            //
            // If this is a node with no inputs and one output, we save
            // this for a second pass, so that the consumer's placement
            // is chosen.
            if (IsGeneratorNode(node)) {
                second_pass.push_back(node);
                continue;
            }

            const std::vector<Device*>* devices;
            Status status = colocation_graph.GetDevicesForNode(node, &devices);
            if (!status.ok()) {
                return AttachDef(
                        errors::InvalidArgument("Cannot assign a device for operation ",
                            node->name(), ": ", status.error_message()),
                        *node);
            }

            // Returns the first device in sorted devices list so we will always
            // choose the same device.
            //
            // TODO(vrv): Factor this assignment out into a pluggable
            // algorithm, so that Placer is responsible for enforcing
            // preconditions and we can experiment with other algorithms when
            // given a choice of devices. Once we have a better idea of the
            // types of heuristics we want to use and the information needed
            // to perform good placement we can add an interface for this.
            int assigned_device = -1;

            // Provide the default, if necessary.
            if (assigned_device == -1) {
                assigned_device = graph_->InternDeviceName((*devices)[0]->name());
            }

            TF_RETURN_IF_ERROR(AssignAndLog(assigned_device, node, &colocation_graph,
                        log_device_placement_));
        }
        // Perform a second pass assignment for those nodes explicitly
        // skipped during the first pass.
        for (Node* node : second_pass) {
            const std::vector<Device*>* devices;
            Status status = colocation_graph.GetDevicesForNode(node, &devices);
            if (!status.ok()) {
                return AttachDef(
                        errors::InvalidArgument("Cannot assign a device for operation ",
                            node->name(), ": ", status.error_message()),
                        *node);
            }

            int assigned_device = -1;

            // Heuristic A application.
            if (IsGeneratorNode(node) && !node->out_edges().empty()) {
            }

            // Provide the default, if necessary.
            if (assigned_device == -1) {
                assigned_device = graph_->InternDeviceName((*devices)[0]->name());
            }

            TF_RETURN_IF_ERROR(AssignAndLog(assigned_device, node, &colocation_graph,
                        log_device_placement_));
        }
        return Status::OK();
    }
} // namespace dlxnet