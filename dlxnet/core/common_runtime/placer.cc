#include "dlxnet/core/common_runtime/placer.h"
#include "dlxnet/core/platform/logging.h"


namespace dlxnet{
    namespace{
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

        // Status AssignAndLog(int assigned_device, Node* node,
        // ColocationGraph* colocation_graph,
        // bool log_device_placement) {
        // node->set_assigned_device_name_index(assigned_device);

        // // Constraint the group of node to the assigned device.
        // TF_RETURN_IF_ERROR(colocation_graph->LimitToAssignedDevice(*node));

        // LogDeviceAssignment(node, log_device_placement);
        // return Status::OK();
        // }
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

        // For each node, assign a device based on the constraints in the disjoint
        // node set.
        std::vector<Node*> second_pass;
        for (Node* node : graph_->op_nodes()) {
        }
    }
} // namespace dlxnet
