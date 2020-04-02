#include "dlxnet/core/kernels/function_ops.h"

namespace dlxnet{
    ArgOp::ArgOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("index", &index_));
    }
    void ArgOp::Compute(OpKernelContext* ctx) {
        auto frame = ctx->call_frame();
        OP_REQUIRES(ctx, frame != nullptr, errors::Internal("no call frame"));
        Tensor val;
        OP_REQUIRES_OK(ctx, frame->GetArg(index_, &val));
        OP_REQUIRES(ctx, val.dtype() == dtype_,
                errors::InvalidArgument("Type mismatch: actual ",
                    DataTypeString(val.dtype()),
                    " vs. expect ", DataTypeString(dtype_)));
        ctx->set_output(0, val);
    }

    RetvalOp::RetvalOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("index", &index_));
    }

    void RetvalOp::Compute(OpKernelContext* ctx) {
        const Tensor& val = ctx->input(0);
        OP_REQUIRES(ctx, val.dtype() == dtype_,
                errors::InvalidArgument("Type mismatch: actual ",
                    DataTypeString(val.dtype()),
                    " vs. expect ", DataTypeString(dtype_)));
        auto frame = ctx->call_frame();
        OP_REQUIRES(ctx, frame != nullptr, errors::Internal("no call frame"));
        OP_REQUIRES_OK(ctx, frame->SetRetval(index_, val));
    }

    REGISTER_SYSTEM_KERNEL_BUILDER(Name(kArgOp).Device(DEVICE_CPU), ArgOp);
    REGISTER_SYSTEM_KERNEL_BUILDER(Name(kRetOp).Device(DEVICE_CPU), RetvalOp);
}


