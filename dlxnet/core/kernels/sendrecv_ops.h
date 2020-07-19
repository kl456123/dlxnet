#ifndef DLXNET_CORE_KERNELS_SENDRECV_OPS_H_
#define DLXNET_CORE_KERNELS_SENDRECV_OPS_H_
#include "dlxnet/core/framework/op_kernel.h"
#include "dlxnet/core/platform/macros.h"

namespace dlxnet{
    class SendOp : public OpKernel {
        public:
            explicit SendOp(OpKernelConstruction* ctx);
            void Compute(OpKernelContext* ctx) override;

        private:
            string key_prefix_;
            Rendezvous::ParsedKey parsed_key_;
            bool hostmem_sendrecv_;

            TF_DISALLOW_COPY_AND_ASSIGN(SendOp);
    };

    class RecvOp : public AsyncOpKernel {
        public:
            explicit RecvOp(OpKernelConstruction* ctx);
            void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

        private:
            string key_prefix_;
            Rendezvous::ParsedKey parsed_key_;
            bool hostmem_sendrecv_;

            TF_DISALLOW_COPY_AND_ASSIGN(RecvOp);
    };
} // namespace dlxnet


#endif
