#include "dlxnet/core/kernels/function_ops.h"

namespace dlxnet{
    ArgOp::ArgOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void ArgOp::Compute(OpKernelContext* ctx) {}

    RetvalOp::RetvalOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void RetvalOp::Compute(OpKernelContext* ctx) {}

    REGISTER_SYSTEM_KERNEL_BUILDER(Name(kArgOp).Device(DEVICE_CPU), ArgOp);
    REGISTER_SYSTEM_KERNEL_BUILDER(Name(kRetOp).Device(DEVICE_CPU), RetvalOp);
}


