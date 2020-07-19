#include "dlxnet/core/framework/memory_types.h"
#include "dlxnet/core/framework/node_def_util.h"
#include "dlxnet/core/framework/op_kernel.h"

namespace dlxnet{
    namespace{
        // Returns the largest endpoint of anything in the name_map.
        int GetTotal(const NameRangeMap& name_map) {
            int total = 0;
            for (const auto& item : name_map) {
                total = std::max(total, item.second.second);
            }
            return total;
        }

        // Fills memory_types for either input or output, setting everything
        // to DEVICE_MEMORY except those args in host_memory_args.  Removes
        // elements of host_memory_args that were used.
        void MemoryTypesHelper(const NameRangeMap& name_map,
                std::vector<string>* host_memory_args,
                MemoryTypeVector* memory_types) {
            // Update args that have been marked as in "HOST_MEMORY".
            size_t keep = 0;
            for (size_t i = 0; i < host_memory_args->size(); ++i) {
                auto iter = name_map.find((*host_memory_args)[i]);
                if (iter != name_map.end()) {
                    for (int j = iter->second.first; j < iter->second.second; ++j) {
                        (*memory_types)[j] = HOST_MEMORY;
                    }
                } else {
                    // (*host_memory_args)[i] not found, save it for the next pass.
                    if (i > keep) (*host_memory_args)[keep] = (*host_memory_args)[i];
                    ++keep;
                }
            }
            host_memory_args->resize(keep);
        }
    } // namespace

    Status MemoryTypesForNode(const OpRegistryInterface* op_registry,
            const DeviceType& device_type, const NodeDef& ndef,
            MemoryTypeVector* inp_mtypes,
            MemoryTypeVector* out_mtypes) {
        // Look up the Op registered for this op name.
        const OpDef* op_def;
        TF_RETURN_IF_ERROR(op_registry->LookUpOpDef(ndef.op(), &op_def));

        // Look up the Kernel registered for this node def.
        const KernelDef* kdef = nullptr;
        Status status =
            FindKernelDef(device_type, ndef, &kdef, nullptr /* kernel_class_name */);

        DataTypeVector inp_dtypes;
        DataTypeVector out_dtypes;
        TF_RETURN_IF_ERROR(
                InOutTypesForNode(ndef, *op_def, &inp_dtypes, &out_dtypes));

        inp_mtypes->clear();
        out_mtypes->clear();

        // For functions (which have no KernelDef) and their gradients, we can only
        // best-effort derive the memory type from the data type. For now, we assume
        // int32 is always on host memory and other types are always on device memory.
        // TODO(zhifengc,phawkins): We should do type inference over function bodies
        // to derive the correct input/output memory types. We should also split
        // host-memory and non host-memory arguments into separate type lists.
        if (!status.ok()) {
        }

        // Gets the input/output names and their corresponding endpoint ranges.
        NameRangeMap inp_names;
        NameRangeMap out_names;
        TF_RETURN_IF_ERROR(NameRangesForNode(ndef, *op_def, &inp_names, &out_names));

        // Now that we know the size, fill with the default 'DEVICE_MEMORY'.
        inp_mtypes->resize(GetTotal(inp_names), DEVICE_MEMORY);
        out_mtypes->resize(GetTotal(out_names), DEVICE_MEMORY);

        // Fills in host memory types based on the kernel def.
        const auto& from_proto = kdef->host_memory_arg();
        std::vector<string> host_memory_args(from_proto.begin(), from_proto.end());
        MemoryTypesHelper(inp_names, &host_memory_args, inp_mtypes);
        MemoryTypesHelper(out_names, &host_memory_args, out_mtypes);
        if (!host_memory_args.empty()) {
            return errors::InvalidArgument(
                    "HostMemory args '", absl::StrJoin(host_memory_args, "', '"),
                    "' not found in OpDef: ", SummarizeOpDef(*op_def));
        }

        CHECK_LE(inp_mtypes->size(), inp_dtypes.size());
        CHECK_LE(out_mtypes->size(), out_dtypes.size());

        // Mark e.g. all resource and string types as host memory.
        for (int i = 0; i < inp_mtypes->size(); ++i) {
            if (DataTypeAlwaysOnHost(inp_dtypes[i])) {
                (*inp_mtypes)[i] = HOST_MEMORY;
            }
        }

        for (int i = 0; i < out_mtypes->size(); ++i) {
            if (DataTypeAlwaysOnHost(out_dtypes[i])) {
                (*out_mtypes)[i] = HOST_MEMORY;
            }
        }

        std::vector<int32> hostmem_attr;
        if (TryGetNodeAttr(ndef, "_input_hostmem", &hostmem_attr)) {
            for (int32 i : hostmem_attr) {
                if (0 <= i && i < inp_mtypes->size()) {
                    (*inp_mtypes)[i] = HOST_MEMORY;
                }
            }
        }
        if (TryGetNodeAttr(ndef, "_output_hostmem", &hostmem_attr)) {
            for (int32 i : hostmem_attr) {
                if (0 <= i && i < out_mtypes->size()) {
                    (*out_mtypes)[i] = HOST_MEMORY;
                }
            }
        }
        return Status::OK();
    }

} // namespace dlxnet
