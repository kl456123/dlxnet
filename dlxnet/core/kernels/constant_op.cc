/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/array_ops.cc.
#include "dlxnet/core/kernels/constant_op.h"
#include "dlxnet/core/framework/types.h"

namespace dlxnet {

    ConstantOp::ConstantOp(OpKernelConstruction* ctx)
        : OpKernel(ctx),
        tensor_(ctx->output_type(0)) {
            const TensorProto* proto = nullptr;
            OP_REQUIRES_OK(ctx, ctx->GetAttr("value", &proto));
            OP_REQUIRES_OK(ctx, ctx->device()->MakeTensorFromProto(
                        *proto, AllocatorAttributes(), &tensor_));
            OP_REQUIRES(
                    ctx, ctx->output_type(0) == tensor_.dtype(),
                    errors::InvalidArgument("Type mismatch between value (",
                        DataTypeString(tensor_.dtype()), ") and dtype (",
                        DataTypeString(ctx->output_type(0)), ")"));
        }

    void ConstantOp::Compute(OpKernelContext* ctx) {
        ctx->set_output(0, tensor_);
        if (TF_PREDICT_FALSE(ctx->track_allocations())) {
            ctx->record_persistent_memory_allocation(tensor_.AllocatedBytes());
        }
    }

    ConstantOp::~ConstantOp() {}

    REGISTER_KERNEL_BUILDER(Name("Const").Device(DEVICE_CPU), ConstantOp);

#define REGISTER_KERNEL(D, TYPE)                                      \
    REGISTER_KERNEL_BUILDER(                                            \
            Name("Const").Device(DEVICE_##D).TypeConstraint<TYPE>("dtype"), \
            ConstantOp);
    REGISTER_KERNEL(GPU, float);
    REGISTER_KERNEL(GPU, double);
    REGISTER_KERNEL(GPU, uint8);
    REGISTER_KERNEL(GPU, int8);
    REGISTER_KERNEL(GPU, bool);
    REGISTER_KERNEL(GPU, int64);
    REGISTER_KERNEL(GPU, uint64);
    REGISTER_KERNEL(GPU, int32);
    REGISTER_KERNEL(GPU, uint32);

}  // namespace dlxnet
