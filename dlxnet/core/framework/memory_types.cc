#include "dlxnet/core/framework/memory_types.h"
#include "dlxnet/core/framework/node_def_util.h"
#include "dlxnet/core/framework/op_kernel.h"

namespace dlxnet{
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
        return Status::OK();
    }

} // namespace dlxnet
