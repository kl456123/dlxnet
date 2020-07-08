#include "dlxnet/core/common_runtime/colocation_graph.h"
#include "dlxnet/core/common_runtime/placer_inspection_required_ops_utils.h"
#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "dlxnet/core/framework/op_kernel.h"

namespace dlxnet{
    namespace{
        // Using absl::StrJoin with lambda does not work in tf-lite builds.
        std::vector<string> DevicesToString(const std::vector<Device*> devices) {
            std::vector<string> v;
            v.reserve(devices.size());
            for (Device* d : devices) {
                v.push_back(d->name());
            }
            return v;
        }

        // Returns ParsedName whose address space (i.e. job, replica, task) identifies
        // the address space directly accessible by the local process. If the address
        // space is fully specified and it is exactly the same as the address space
        // of a device, then all kernels of that device should be registered in the
        // local process.
        static const DeviceNameUtils::ParsedName LocalAddressSpec(
                const Device* client_device, const Device* default_local_device) {
            if (client_device != nullptr) {
                return DeviceNameUtils::AddressSpace(client_device->parsed_name());
            }

            if (default_local_device != nullptr) {
                return DeviceNameUtils::AddressSpace(default_local_device->parsed_name());
            }

            // TODO(b/139617593) Return the name of the first local device in device_set_
            // once we can trust the output of Device::IsLocal().
            return DeviceNameUtils::ParsedName();
        }
    } // namespace

    Status Member::SetAssignedDeviceName(const string& device_name) {
        if (DeviceNameUtils::HasSomeDetails(requested_device_name_)) {
            return errors::Internal(
                    "Setting assigned device name when there is a requested device set "
                    "is unsupported");
        }
        if (!DeviceNameUtils::ParseFullName(device_name, &assigned_device_name_)) {
            return errors::Internal("Malformed assigned device '", device_name, "'");
        }
        // Set requested device to assigned_device to maintain the invariant that
        // requested is a specialization of assigned.
        requested_device_name_ = assigned_device_name_;
        return Status::OK();
    }

    int Member::FindRoot(const std::vector<Member>& tree, int node_id) {
        const Member& member = tree[node_id];
        if (member.parent_ == node_id) {
            return member.parent_;
        }
        return FindRoot(tree, member.parent_);
    }

    Status Member::SetParentAndSupportedDevices(
            const Node& node, const std::vector<DeviceType>& types,
            const DeviceNameUtils::ParsedName* local_address_spec) {
        int id = node.id();
        if (id < 0) {
            return errors::Internal("Placer should not be creating a Member for node: ",
                    node.DebugString());
        }
        parent_ = id;
        return SupportedDeviceTypesForNode(
                types, node.def(), &supported_device_types_, local_address_spec);
    }

    Status Member::SetResourceDeviceName(const Node& node) {
        if (DeviceNameUtils::HasSomeDetails(requested_device_name_)) {
            return errors::Internal(
                    "Setting resource device name when there is a requested device set "
                    "is unsupported");
        }

        if (!DeviceNameUtils::ParseFullName(node.requested_device(),
                    &resource_device_name_)) {
            return errors::InvalidArgument("Malformed device specification '",
                    node.requested_device(),
                    "' in node: ", node.DebugString());
        }

        // Set requested device to resource device to maintain the invariant that
        // requested is a specialization of resource.
        requested_device_name_ = resource_device_name_;
        return Status::OK();
    }

    Status Member::SetRequestedDeviceName(const Node& node) {
        if (DeviceNameUtils::HasSomeDetails(assigned_device_name_)) {
            return errors::Internal(
                    "Setting requested device name when there is an assigned device set "
                    "is unsupported");
        }
        if (DeviceNameUtils::HasSomeDetails(resource_device_name_)) {
            return errors::Internal(
                    "Setting requested device name when there is a resource device set "
                    "is unsupported");
        }
        if (!DeviceNameUtils::ParseFullName(node.requested_device(),
                    &requested_device_name_)) {
            return errors::InvalidArgument("Malformed device specification '",
                    node.requested_device(),
                    "' in node: ", node.DebugString());
        }
        return Status::OK();
    }

    Status Member::AssignDevice(const Node& node) {
        if (node.assigned_device_name_index() == assigned_device_name_index_) {
            return Status::OK();
        }

        DeviceNameUtils::ParsedName parsed;
        DeviceNameUtils::ParseFullName(node.assigned_device_name(), &parsed);
        Status s = DeviceNameUtils::MergeDevNames(&assigned_device_name_, parsed);
        if (!s.ok()) {
            return errors::Internal(
                    "Constraining by assigned device should not cause an error. Original "
                    "root's assigned device name: ",
                    DeviceNameUtils::ParsedNameToString(assigned_device_name_),
                    " node's assigned device name \"", node.assigned_device_name(),
                    ". Error: ", s.error_message());
        }
        s = DeviceNameUtils::MergeOverrideDevNames(&resource_device_name_, parsed);
        if (!s.ok()) {
            return errors::Internal(
                    "Constraining by assigned device should not cause an error. Original "
                    "root's resource device name: ",
                    DeviceNameUtils::ParsedNameToString(resource_device_name_),
                    " node's assigned device name \"", node.assigned_device_name(),
                    ". Error: ", s.error_message());
        }
        s = DeviceNameUtils::MergeOverrideDevNames(&requested_device_name_, parsed);
        if (!s.ok()) {
            return errors::Internal(
                    "Constraining by assigned device should not cause an error. Original "
                    "root's requested device name: \"",
                    DeviceNameUtils::ParsedNameToString(requested_device_name_),
                    "\", node's assigned device name \"", node.assigned_device_name(),
                    "\". Error: ", s.error_message());
        }

        assigned_device_name_index_ = node.assigned_device_name_index();
        // Clear cached possible_devices, if any.
        possible_devices_.clear();
        return Status::OK();
    }

    string Member::DebugString() const {
        return absl::StrCat(
                "Member(assigned_device_name_index_=", assigned_device_name_index_,
                " requested_device_name_='",
                DeviceNameUtils::ParsedNameToString(requested_device_name_),
                "' assigned_device_name_='",
                DeviceNameUtils::ParsedNameToString(assigned_device_name_),
                "' resource_device_name_='",
                DeviceNameUtils::ParsedNameToString(resource_device_name_),
                "' supported_device_types_=[",
                // absl::StrJoin(DeviceTypeAndPriorityToString(supported_device_types_),
                // ", "),
                "] possible_devices_=[",
                absl::StrJoin(DevicesToString(possible_devices_), ", "), "]");
    }

    // tree is non-const because we can change some `parent` pointers in some
    // members for more efficient future lookups. The vector itself is not
    // changed.
    int Member::FindAndUpdateRoot(std::vector<Member>* tree, int node_id) {
        Member& member = (*tree)[node_id];
        if (member.parent_ == node_id) {
            // member.parent is the root of this disjoint tree.  Do nothing.
        } else {
            member.parent_ = FindAndUpdateRoot(tree, member.parent_);
        }
        // Now it is guaranteed that member.parent is the root of this disjoint
        // tree.
        return member.parent_;
    }

    ColocationGraph::ColocationGraph(const Graph* graph, const FunctionStack& stack,
            const FunctionLibraryDefinition* flib_def,
            const DeviceSet* device_set,
            const Device* default_local_device,
            bool allow_soft_placement,
            bool log_device_placement)
        : graph_(*graph),
        stack_(stack),
        flib_def_(*flib_def),
        // inspecting_placer_(stack, flib_def, device_set, default_local_device,
        // allow_soft_placement, log_device_placement),
        // inspection_required_checker_(graph, flib_def),
        device_set_(*device_set),
        device_types_(device_set->PrioritizedDeviceTypeList()),
        local_address_spec_(
                LocalAddressSpec(device_set->client_device(), default_local_device)),
        default_local_device_(default_local_device),
        allow_soft_placement_(allow_soft_placement),
        log_device_placement_(log_device_placement) {
            members_.resize(graph_.num_node_ids());
        }

    Status ColocationGraph::Initialize() {
        TF_RETURN_IF_ERROR(InitializeMembers());

        // std::unordered_set<Node*> inspection_required;
        // TF_RETURN_IF_ERROR(ColocateResourceAndRefEdges(&inspection_required));
        // TF_RETURN_IF_ERROR(AddInspectionConstraints(inspection_required));
        // TF_RETURN_IF_ERROR(ColocateAllNodes());

        for (Node* node : graph_.op_nodes()) {
            int root_id = FindAndUpdateRoot(node->id());
            // members_[root_id].MaybeExcludeXlaDevices();
        }

        return Status::OK();
    }

    Status ColocationGraph::LimitToAssignedDevice(const Node& node) {
        if (node.assigned_device_name_index() < 0) {
            return errors::Internal(
                    "Expected an assigned node as argument to LimitToAssignedDevice but "
                    "got: ",
                    node.DebugString());
        }
        int root = FindAndUpdateRoot(node.id());
        Member& root_member = members_[root];
        return root_member.AssignDevice(node);
    }



    Status ColocationGraph::GetDevicesForNode(
            Node* node, const std::vector<Device*>** possible_devices) {
        *possible_devices = nullptr;
        const int node_root = FindAndUpdateRoot(node->id());
        if (!members_[node_root].possible_devices().empty()) {
            *possible_devices = &members_[node_root].possible_devices();
            return Status::OK();
        }

        Member& root_member = members_[node_root];

        // We have not yet computed the possible devices for the
        // colocated node set containing 'node', so we do so now using the
        // constraints on the root node.

        // "devices" will contain the set of feasible placements for the
        // colocated node set containing 'node'.
        // NOTE: Basing possible device computation on requested device name
        // is guaranteed to respect the assigned and resource device names because
        // requested device is always a specialization of both.
        std::vector<Device*> devices;
        if (DeviceNameUtils::HasSomeDetails(root_member.requested_device_name())) {
            // The root node has a (possibly partial) device
            // specification, so enumerate the physical devices that
            // conform to it.
            device_set_.FindMatchingDevices(root_member.requested_device_name(),
                    &devices);

            if (!devices.empty()) {
                // Filter devices into those that are compatible with the root
                // node (and its children).
                devices = FilterSupportedDevices(
                        devices, root_member.supported_device_types(), default_local_device_);
            }

            // // Perform soft placement if allow_soft_placement_ is set.
            // if (devices.empty() && allow_soft_placement_) {
            // GetSoftDeviceCandidates(*node, root_member, node_root, &devices);
            // }
            if (devices.empty()) {
                // Return an error when a physical device that matches an explicit
                // device specification is not found. This ensures that we don't
                // assign a node to GPU when the user wanted to force it on CPU.
                string debug_info = DebugInfo(node_root);

                DeviceNameUtils::ParsedName specified_device_name;
                if (DeviceNameUtils::ParseFullName(node->requested_device(),
                            &specified_device_name)&&
                        specified_device_name == root_member.requested_device_name()) {
                    // The specified device and merged set device match, and
                    // will appear in the GraphDef (for debugging), so just
                    // print the specified device.
                    std::vector<Device*> devices_matching_nodedef;
                    device_set_.FindMatchingDevices(specified_device_name,
                            &devices_matching_nodedef);
                    if (devices_matching_nodedef.empty()) {
                        // Sometimes it is almost impossible to understand the problem
                        // without a list of available devices.
                        std::vector<string> device_names;
                        for (const Device* device : device_set_.devices()) {
                            device_names.push_back(device->name());
                        }
                        std::sort(device_names.begin(), device_names.end());

                        string gpu_msg = "";
                        if (absl::AsciiStrToLower(specified_device_name.type) == "gpu") {
                            gpu_msg =
                                " The requested device appears to be a GPU, but CUDA is not "
                                "enabled.";
                        }

                        return errors::InvalidArgument(
                                errors::FormatNodeNameForError(node->name()),
                                " was explicitly assigned to ", node->requested_device(),
                                " but available devices are [ ",
                                absl::StrJoin(device_names, ", "), " ]. Make sure ",
                                "the device specification refers to a valid device.", gpu_msg);
                    }else if(specified_device_name.has_type){
                        return errors::InvalidArgument(
                                "Could not satisfy explicit device specification '",
                                node->requested_device(), "' because no supported kernel for ",
                                specified_device_name.type, " devices is available.", debug_info,
                                "\nOp: ", node->type_string(),
                                "\nNode attrs: ", node->attrs().DebugString(),
                                "\nRegistered kernels:\n",
                                KernelsRegisteredForOp(node->type_string()));
                    }else{
                        return errors::InvalidArgument(
                                "Could not satisfy explicit device specification '",
                                node->requested_device(), debug_info);
                    }
                }else{
                    // The specified device may be a valid device but the
                    // merged set device is different, so print both.
                    // TODO(b/129057603): There are many possibilities at this point.
                    // Provide good error messages.
                    return errors::InvalidArgument(
                            "Could not satisfy explicit device specification '",
                            node->requested_device(), "' because the node ",
                            // errors::FormatColocationNodeForError(node->name()),
                            " was colocated with a group of nodes that ",
                            "required incompatible device '",
                            DeviceNameUtils::ParsedNameToString(
                                root_member.requested_device_name()),
                            "'. All available devices [",
                            absl::StrJoin(DevicesToString(device_set_.devices()), ", "), "]. ",
                            debug_info);
                }
            }// error happened here
        }else{
            // The device is completely unspecified, so enumerate the devices that
            // support all of the nodes in the set.
            if (device_set_.devices().empty()) {
                return errors::Internal("No devices are registered");
            }
            devices = FilterSupportedDevices(device_set_.devices(),
                    root_member.supported_device_types(),
                    default_local_device_);

            if (devices.empty()) {
                return errors::InvalidArgument(
                        "Node had no OpKernel registered to support this operation: ",
                        "Operation was ", node->type_string(), " and inputs were [",
                        DataTypeVectorString(node->input_types()), "].\n",
                        DebugInfo(node_root));
            }
        }

        // Cache the result of the possible devices for this node group.
        root_member.set_possible_devices(std::move(devices));
        *possible_devices = &root_member.possible_devices();
    }

    // Returns a list of devices having type in supported_device_types.  The
    // returned list is sorted by preferred type (higher numeric type is preferred).
    /*static*/ std::vector<Device*> ColocationGraph::FilterSupportedDevices(
            const std::vector<Device*>& devices,
            const PrioritizedDeviceTypeVector& supported_device_types,
            const Device* default_local_device) {
        Device* filtered_default_device = nullptr;
        std::vector<std::pair<Device*, int32>> prioritized_filtered_devices;
        for (const auto& supported_device_type : supported_device_types) {
            for (Device* device : devices) {
                if (DeviceType(device->attributes().device_type()) ==
                        supported_device_type.first) {
                    if (default_local_device &&
                            (device == default_local_device ||
                             // TODO(nareshmodi, fishx): At times the device pointer in the
                             // device set is different to the one passed in as the default
                             // device. Figure out why this might be.
                             device->name() == default_local_device->name())) {
                        filtered_default_device = device;
                    } else {
                        prioritized_filtered_devices.emplace_back(
                                device, supported_device_type.second);
                    }
                }
            }
        }

        auto device_sort = [](const std::pair<Device*, int32>& a,
                const std::pair<Device*, int32>& b) {
            if (a.second != b.second) {
                return a.second > b.second;
            }

            auto a_priority =
                DeviceSet::DeviceTypeOrder(DeviceType(a.first->device_type()));
            auto b_priority =
                DeviceSet::DeviceTypeOrder(DeviceType(b.first->device_type()));
            // First sort by prioritized device type (higher is preferred) and
            // then by device name (lexicographically).
            if (a_priority != b_priority) {
                return a_priority > b_priority;
            }
            return StringPiece(a.first->name()) < StringPiece(b.first->name());
        };
        std::sort(prioritized_filtered_devices.begin(),
                prioritized_filtered_devices.end(), device_sort);

        std::vector<Device*> filtered_devices;
        if (filtered_default_device != nullptr) {
            filtered_devices.emplace_back(filtered_default_device);
        }
        for (const auto& prioritized_filtered_device : prioritized_filtered_devices) {
            filtered_devices.push_back(prioritized_filtered_device.first);
        }
        return filtered_devices;
    }

    // Returns debugging info for the node referred to by 'node_root'.
    string ColocationGraph::DebugInfo(const int node_root) const {
        string text(
                "\nColocation Debug Info:\n"
                "Colocation group had the following types and supported devices: ");
        return text;
    }

    Status ColocationGraph::InitializeMemberWithAssignedDevice(
            const string& assigned_device_name, const string& node_type,
            Member* member) {
        // This node has already been assigned to a device, so we
        // respect this placement, after sanity-checking it.
        // NOTE: Since any assignment must have been performed by
        // the TensorFlow runtime, we consider errors in this branch to
        // be INTERNAL.
        TF_RETURN_IF_ERROR(member->SetAssignedDeviceName(assigned_device_name));

        // Since assigned device must be a full specification, do extra checks.
        const Device* assigned_device =
            device_set_.FindDeviceByName(assigned_device_name);
        if (assigned_device == nullptr) {
            // TODO(b/129295848, b/122851476): Remove the bit about cross-host function
            // calls when they are supported.
            return errors::Internal(
                    "Assigned device '", assigned_device_name,
                    "' does not match any device. This error can happen when one attempts "
                    "to run a tf.function with resource inputs residing on remote devices. "
                    "This use case is currently not supported. Here are the devices "
                    "available on this machine: [",
                    absl::StrJoin(DevicesToString(device_set_.devices()), ", "), "].",
                    "If you are seeing this error when running using a tf.Session, set "
                    "experimental.share_cluster_devices_in_session to true in the "
                    "tf.ConfigProto.");
        }
        for (const auto& d : member->supported_device_types()) {
            if (DeviceType(assigned_device->attributes().device_type()) == d.first) {
                return Status::OK();
            }
        }

        return errors::Internal("Assigned device '", assigned_device_name,
                "' does not have registered OpKernel support "
                "for ",
                node_type);
    }

    Status ColocationGraph::InitializeMembers() {
        for (Node* node : graph_.op_nodes()) {
            Status status = InitializeMember(*node, &members_[node->id()]);
            if (!status.ok()) {
                return AttachDef(status, *node);
            }
        }
        return Status::OK();
    }

    Status ColocationGraph::InitializeMember(const Node& node, Member* member) {
        TF_RETURN_IF_ERROR(member->SetParentAndSupportedDevices(
                    node, device_types_, &local_address_spec_));
        if (node.has_assigned_device_name()) {
            TF_RETURN_IF_ERROR(InitializeMemberWithAssignedDevice(
                        node.assigned_device_name(), node.type_string(), member));
        } else {
            // This node has not yet been assigned to a device, so we
            // calculate any constraints due to the set of registered
            // kernels and any (partial) user-provided device specification
            // in the NodeDef.

            // If no kernels are registered for this op type, fail with an error.
            if (member->supported_device_types().empty()) {
                std::set<string> registered_device_types;
                for (Device* d : device_set_.devices()) {
                    registered_device_types.insert(d->device_type());
                }
                return errors::InvalidArgument(
                        "No OpKernel was registered to support Op '", node.type_string(),
                        "' used by ", errors::FormatNodeNameForError(node.name()),
                        "with these attrs: [", node.attrs().DebugString(),
                        "]\n"
                        "Registered devices: [",
                        absl::StrJoin(registered_device_types, ", "), "]\n",
                        "Registered kernels:\n", KernelsRegisteredForOp(node.type_string()));
            }

            // If the NodeDef contains a device, then we interpret it as a
            // (partial) device specification.
            if (!node.requested_device().empty()) {
                // if (IsRefOrResourceGeneratorNode(node)) {
                // Treat requested device on resource generating nodes as assigned
                // device so that we don't override it.
                // TF_RETURN_IF_ERROR(member->SetResourceDeviceName(node));
                // } else {
                // The user has specified a device in the NodeDef, try to find a
                // valid device matching their specification in the set of
                // devices.
                // NOTE: The full name may specify a device that is not in
                // n.supported_device_types(), but we check that in AssignDevice().
                TF_RETURN_IF_ERROR(member->SetRequestedDeviceName(node));
                // }
            }
        }
        return Status::OK();
    }
} // namespace dlxnet
