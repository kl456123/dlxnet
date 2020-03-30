#ifndef DLXNET_CORE_KERNELS_FUNCTION_OPS_H_
#define DLXNET_CORE_KERNELS_FUNCTION_OPS_H_
#include "dlxnet/core/framework/function.h"
#include "dlxnet/core/framework/op_kernel.h"

namespace dlxnet{
    // used as registered kernel name
    static const char* const kArgOp = FunctionLibraryDefinition::kArgOp;
    static const char* const kDeviceArgOp = FunctionLibraryDefinition::kDeviceArgOp;
    static const char* const kRetOp = FunctionLibraryDefinition::kRetOp;
    static const char* const kDeviceRetOp = FunctionLibraryDefinition::kDeviceRetOp;

    // function op used to get arg from input and set return val in output
    class ArgOp : public OpKernel {
        public:
            explicit ArgOp(OpKernelConstruction* ctx);

            void Compute(OpKernelContext* ctx) override;

            bool IsExpensive() override { return false; }

        private:
            int index_;
            DataType dtype_;

            TF_DISALLOW_COPY_AND_ASSIGN(ArgOp);
    };

    class RetvalOp : public OpKernel {
        public:
            explicit RetvalOp(OpKernelConstruction* ctx);

            void Compute(OpKernelContext* ctx) override;

            bool IsExpensive() override { return false; }

        private:
            int index_;
            DataType dtype_;

            TF_DISALLOW_COPY_AND_ASSIGN(RetvalOp);
    };
}


#endif
