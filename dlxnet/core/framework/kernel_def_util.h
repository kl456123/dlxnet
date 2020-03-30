#ifndef DLXNET_CORE_FRAMEWORK_KERNEL_DEF_UTIL_H_
#define DLXNET_CORE_FRAMEWORK_KERNEL_DEF_UTIL_H_
#include "dlxnet/core/framework/kernel_def.pb.h"
#include "dlxnet/core/framework/node_def_util.h"

namespace dlxnet{
    // Returns whether the attrs satisfy the constraints in the kernel_def. Returns
    // an error if attrs in kernel_def are not found, or have a mismatching type.
    Status KernelAttrsMatch(const KernelDef& kernel_def, AttrSlice attrs,
            bool* match);
}


#endif

