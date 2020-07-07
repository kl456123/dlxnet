#ifndef DLXNET_CORE_COMMON_RUNTIME_COLOCATION_GRAPH_H_
#define DLXNET_CORE_COMMON_RUNTIME_COLOCATION_GRAPH_H_
#include "dlxnet/core/platform/logging.h"
#include "dlxnet/core/graph/graph.h"
#include "dlxnet/core/framework/function.h"
#include "dlxnet/core/common_runtime/device.h"
#include "dlxnet/core/common_runtime/device_set.h"
#include "dlxnet/core/common_runtime/placer_inspection_required_ops_utils.h"
#include "dlxnet/core/lib/core/status.h"
#include "dlxnet/core/util/device_name_utils.h"
#include "dlxnet/core/framework/types.h"

namespace dlxnet{
    // Represents a node in the disjoint node forest and the
    // accumulated constraints on the device used by that node.
    class Member {
        public:
            Member() = default;

            const DeviceNameUtils::ParsedName& requested_device_name() const {
                return requested_device_name_;
            }

            Status SetAssignedDeviceName(const string& device_name);
            Status SetResourceDeviceName(const Node& node);
            Status SetRequestedDeviceName(const Node& node);

            Status AssignDevice(const Node& node);

            void set_possible_devices(std::vector<Device*>&& devices) {
                possible_devices_ = devices;
            }
            const std::vector<Device*>& possible_devices() { return possible_devices_; }

            const PrioritizedDeviceTypeVector& supported_device_types() const {
                return supported_device_types_;
            }

            string DebugString() const;



            static int FindAndUpdateRoot(std::vector<Member>* tree, int node_id);

        private:
            // The id of the node that is the parent of this one, or its own
            // id if it is a root. parent <= 0 indicates that this member is invalid.
            int parent_ = -1;

            // A proxy for the depth of the tree that is used to prefer
            // connecting smaller trees to larger trees when merging disjoint
            // sets.
            int rank_ = 0;

            // Once colocation groups have been formed, the Placer starts actually
            // choosing devices. All nodes in a group must be assigned to the same
            // device. Once we assigned the first device to some node in this group,
            // we set assigned_device_name_index to this device name's index in the
            // graph.
            // The `*_device_name_` fields will contain the parsed name of this device
            // and `possible_devices`, if computed, will contain just this device.
            // `assigned_device_name_index` is an optimization to avoid parsing and
            // comparing device names. The value of -1 signals that a single device
            // has not been chosen yet.
            int assigned_device_name_index_ = -1;

            // The merged form of the device requested for this node, with those of all of
            // its children. requested_device_name_ is always kept a specialization (i.e.
            // DeviceNameUtils::IsSpecification) of assigned_device_name_. When no device
            // is requested, this field is set to assigned_device_name_.  As a
            // specialization of assigned_device_name_, requested_device_name_ represents
            // the most specific form of all assigned and requested devices of this node
            // and its children, if this node is a root. requested_device_name_ is used
            // to finally select devices for nodes.  We can override requested devices due
            // to resource colocation constraints but not assigned devices (unless soft
            // placement is on).
            // INVARIANT: requested_device_name_ is always kept a
            // DeviceNameUtils::IsSpecification of assigned_device_name_ and
            // resource_device_name_. This makes requested_device_name_ the "accumulation
            // of all wishes" about the device.
            DeviceNameUtils::ParsedName requested_device_name_;

            // The merged form of the device assigned for this node, with
            // those of all of its children.
            // This field is used to raise errors due to unsatisfiable constraints.
            // Can be a partial specification.
            DeviceNameUtils::ParsedName assigned_device_name_;

            // The merged form of the requested resource device assigned for this node,
            // with those of all of its children.
            // This field is used to raise errors due to unsatisfiable constraints.
            // Can be a partial specification.
            // resource_device_name_ is initialized with user-requested device on nodes
            // producing resources, e.g. VarHandleOp.
            // For historical reasons, with soft placement enabled, Placer can "move"
            // resources (place resource producing ops on a device different from what
            // the user explicitly requested) when the colocation group of a resource
            // producing op contains ops that are not supported on the user-requested
            // resource device. A classic example of this is a sparse optimizer (only
            // supported on CPU) used on a GPU variable. In this case, the whole group
            // will be assigned to some device supported by all ops in the colocation
            // group. This is a surprising and unfortunate behavior because:
            //   1. Since soft_placement is on by default, users don't know that their
            //   variables are created on a different device than what they requested.
            //   Among other things, this can lead to surprising poor performance.
            //   2. Eager runtime cannot "move" resources. The same code can "work" when
            //   wrapped in tf.function but will fail when run eagerly.
            //   3. Extra complexity here to preserve these resource moving capabilities.
            DeviceNameUtils::ParsedName resource_device_name_;

            // The intersection of all device types supported by this node,
            // and those of all of its children, in priority order
            // of the preferred device.
            // It is possible that supported_device_types_ has an empty intersection with
            // requested/assigned/resource devices. We could have detected such cases
            // as soon as they happen and raise an error. Instead, for historical reasons,
            // we leave such error detection to the final device picking stage.
            PrioritizedDeviceTypeVector supported_device_types_;

            // If this node is a root, stores a list of Devices to which this node
            // and all of its children can be assigned.
            // `possible_devices` is empty if they have not yet been computed.
            std::vector<Device*> possible_devices_;
    };
    // This class maintains the connected components of a colocation
    // constraint graph, and uses this information to assign a satisfying
    // device placement to the nodes of the graph.
    //
    // This implementation uses the Union-Find algorithm to efficiently maintain the
    // connected components and incrementally adds edges via
    // ColocationGraph::ColocateNodes() invocations.
    //
    // ColocationGraph does not assign any devices to graph nodes. The
    // `log_device_placement` argument is used to log messages when requested
    // device is ignored.
    class ColocationGraph {
        public:
            // graph, flib_def, and device_set must not be null and must outlive
            // this ColocationGraph. default_local_device can be null. If not, must
            // outlive this.
            ColocationGraph(const Graph* graph, const FunctionStack& stack,
                    const FunctionLibraryDefinition* flib_def,
                    const DeviceSet* device_set,
                    const Device* default_local_device, bool allow_soft_placement,
                    bool log_device_placement);

            Status Initialize();

            // Returns the root node of the disjoint tree to which the node with the
            // given id is connected.
            // Updates the internal pointers so that future calls will returns faster.
            int FindAndUpdateRoot(int node_id) {
                return Member::FindAndUpdateRoot(&members_, node_id);
            }

            // Limit the group containing `node` to the device specifications in
            // `devices`.
            // Status LimitToPossibleDevices(const Node& node,
            // const PossibleDevices& devices);

            // Limits the possible devices of `node`'s colocation group to the device
            // to which `node` is assigned. This makes sure that all nodes in this
            // colocation group will be assigned to the same device. Without this
            // explicit restriction, heuristics can choose a different possible device
            // for other nodes in the group.
            Status LimitToAssignedDevice(const Node& node);

            // For the given node, subject to the constraints previously given
            // to this ColocationGraph, set its assigned_device_name. Returns OK
            // if a satisfying device can be found, otherwise an error.
            //
            // Note: This method returns a pointer to a field within members_.
            // The caller must not use the returned pointer after there is any possibility
            // that the members_[i].possible_devices field has been modified.
            Status GetDevicesForNode(Node* node,
                    const std::vector<Device*>** possible_devices);

            // Returns debugging info for the node referred to by 'node_root'.
            string DebugInfo(const int node_root) const;

            string DebugString() const;

            // Returns a list of devices having type in supported_device_types.  The
            // returned list is sorted by preferred type (higher numeric type is
            // preferred).
            static std::vector<Device*> FilterSupportedDevices(
                    const std::vector<Device*>& devices,
                    const PrioritizedDeviceTypeVector& supported_device_types,
                    const Device* default_local_device);

        private:
            const Graph& graph_;
            const FunctionStack stack_;
            const FunctionLibraryDefinition& flib_def_;
            std::vector<Member> members_;
            // InspectingPlacer inspecting_placer_;
            // PlacerInspectionRequiredOpChecker inspection_required_checker_;
            const DeviceSet& device_set_;
            const std::vector<DeviceType> device_types_;
            const DeviceNameUtils::ParsedName local_address_spec_;
            const Device* default_local_device_;
            const bool allow_soft_placement_;
            const bool log_device_placement_;

            TF_DISALLOW_COPY_AND_ASSIGN(ColocationGraph);
    };
} // namespace dlxnet


#endif
