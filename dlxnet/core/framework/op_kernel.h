#ifndef DLXNET_CORE_FRAMEWORK_OP_KERNEL_H_
#define DLXNET_CORE_FRAMEWORK_OP_KERNEL_H_
#include <functional>


namespace dlxnet{
    class OpKernelConstruction;
    class OpKernelContext;
    class OpKernel{
        public:
            virtual void Compute(OpKernelContext* context);
    };

    class AsyncOpKernel:public OpKernel{
        public:
            typedef std::function<void()> DoneCallback;
            virtual void ComputeAsync(OpKernelContext* context, DoneCallback done) = 0;
            void Compute(OpKernelContext* context) override;
    };


    class OpKernelConstruction{
    };


    class OpKernelContext{
    };

}


#endif
