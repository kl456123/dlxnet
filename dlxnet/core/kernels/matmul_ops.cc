
#define EIGEN_USE_THREADS
#include "dlxnet/core/kernels/matmul_ops.h"
#include "dlxnet/core/framework/types.h"
#include "dlxnet/core/framework/register_types.h"


namespace dlxnet{
    typedef Eigen::ThreadPoolDevice CPUDevice;

    template<typename Device, typename T>
        class MatMulOp: public OpKernel{
            public:
                explicit MatMulOp(OpKernelConstruction* ctx)
                    :OpKernel(ctx){
                        OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
                        OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
                    }

                void Compute(OpKernelContext* ctx) override {
                    const Tensor& a = ctx->input(0);
                    const Tensor& b = ctx->input(1);

                    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
                    dim_pair[0].first = transpose_a_ ? 0 : 1;
                    dim_pair[0].second = transpose_b_ ? 1 : 0;

                    OP_REQUIRES(
                            ctx, a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
                            errors::InvalidArgument(
                                "Matrix size-incompatible: In[0]: ", a.shape().DebugString(),
                                ", In[1]: ", b.shape().DebugString()));
                    int a_dim_remaining = 1 - dim_pair[0].first;
                    int b_dim_remaining = 1 - dim_pair[0].second;

                    TensorShape out_shape(
                            {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
                    Tensor* out = nullptr;
                    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

                    if(out->NumElements()==0){
                        // If a has shape [0, x] or b has shape [x, 0], the output shape
                        // is a 0-element matrix, so there is nothing to do.
                        return ;
                    }
                    functor::MatMulFunctor<Device, T>()(ctx->eigen_device<Device>(),
                                            out->matrix<T>(), a.matrix<T>(),
                                            b.matrix<T>(), dim_pair);
                }
            private:
                bool transpose_a_;
                bool transpose_b_;
        };

    namespace functor{
        template <typename T>
            struct MatMulFunctor<CPUDevice, T> {
                void operator()(
                        const CPUDevice& d, typename MatMulTypes<T>::out_type out,
                        typename MatMulTypes<T>::in_type in0,
                        typename MatMulTypes<T>::in_type in1,
                        const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair) {
                    MatMul<CPUDevice>(d, out, in0, in1, dim_pair);
                }
            };
    }//namespace functor

#define REGISTER_CPU(T)                                             \
    REGISTER_KERNEL_BUILDER(                                          \
            Name("MatMul").Device(DEVICE_CPU).TypeConstraint<T>("T"),     \
            MatMulOp<CPUDevice, T>)

    TF_CALL_float(REGISTER_CPU);
    TF_CALL_double(REGISTER_CPU);
    TF_CALL_int32(REGISTER_CPU);
}// namespace dlxnet
