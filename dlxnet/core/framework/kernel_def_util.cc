#include "dlxnet/core/framework/kernel_def_util.h"


namespace dlxnet{
    Status KernelAttrsMatch(const KernelDef& kernel_def, AttrSlice attrs,
            bool* match) {
        *match = true;
        return Status::OK();
    }
}
