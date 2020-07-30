
#define EIGEN_USE_THREADS
#include "dlxnet/core/kernels/matmul_ops.h"
#include "dlxnet/core/framework/types.h"
#include "dlxnet/core/framework/register_types.h"
#include "dlxnet/core/kernels/gpu_utils.h"


namespace dlxnet{
    typedef Eigen::ThreadPoolDevice CPUDevice;
    // typedef Eigen::GpuDevice GPUDevice;
    class GPUDevice{};

    template <typename Device, typename T>
        struct LaunchMatMul;

    template <typename Device, typename T>
        struct LaunchMatMulBase {
            static void launch(
                    OpKernelContext* ctx, const Tensor& a, const Tensor& b,
                    const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
                    Tensor* out) {
                functor::MatMulFunctor<Device, T>()(ctx->eigen_device<Device>(),
                        out->matrix<T>(), a.matrix<T>(),
                        b.matrix<T>(), dim_pair);
            }
        };
    template <typename T>
        struct LaunchMatMulCPU : LaunchMatMulBase<CPUDevice, T> {};
    template <typename T>
        struct LaunchMatMul<CPUDevice, T> : public LaunchMatMulCPU<T> {};

    template <typename T>
        struct LaunchMatMul<GPUDevice, T> {
            static void launch(
                    OpKernelContext* ctx, const Tensor& a, const Tensor& b,
                    const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
                    Tensor* out) {
                const uint64 m = a.dim_size(1 - dim_pair[0].first);
                const uint64 k = a.dim_size(dim_pair[0].first);
                const uint64 n = b.dim_size(1 - dim_pair[0].second);
                bool transpose_a = dim_pair[0].first == 0;
                bool transpose_b = dim_pair[0].second == 1;

                auto* stream = ctx->op_device_context()->stream();
                OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

                auto a_ptr = AsDeviceMemory(a.template flat<T>().data(),
                        a.template flat<T>().size());
                auto b_ptr = AsDeviceMemory(b.template flat<T>().data(),
                        b.template flat<T>().size());
                auto c_ptr = AsDeviceMemory(out->template flat<T>().data(),
                        out->template flat<T>().size());
                auto alpha = static_cast<T>(1.0);
                auto beta = static_cast<T>(0.0);
                // Use C' = B' x A' (' stands for transpose)
                // bool blas_launch_status =
                // stream
                // ->ThenLaunch(transpose_b, transpose_a, n, m, k,
                // 1.0f, b_ptr, transpose_b ? k : n, a_ptr,
                // transpose_a ? m : k, 0.0f, &c_ptr, n)
                // .ok();

                // create kernel
                auto executor = stream->parent();
                se::MultiKernelLoaderSpec kernel_loader_spec(6);
                const string kernel_fn = "../dlxnet/example/cl/matmul.ocl";
                kernel_loader_spec.AddOpenCLTextOnDisk(kernel_fn,
                        "matmul");
                using KernelType = se::TypedKernel<se::DeviceMemory<T>*,
                      se::DeviceMemory<T> *, se::DeviceMemory<T> *, uint64, uint64, uint64>;
                KernelType kernel(executor);
                auto status = executor->GetKernel(kernel_loader_spec, &kernel);
                if(!status.ok()){
                    ctx->SetStatus(errors::Internal("Get Kernel Failed : ", kernel_fn));
                }
                const uint64 N = m*n;

                bool blas_launch_status = (*stream)
                    .ThenLaunch(se::ThreadDim(), se::BlockDim(N), kernel,
                            &a_ptr, &b_ptr, &c_ptr, m, n, k)
                    .BlockHostUntilDone().ok();
                // bool blas_launch_status;
                if (!blas_launch_status) {
                    ctx->SetStatus(errors::Internal(
                                "Blas GEMM launch failed : a.shape=(", a.dim_size(0), ", ",
                                a.dim_size(1), "), b.shape=(", b.dim_size(0), ", ", b.dim_size(1),
                                "), m=", m, ", n=", n, ", k=", k));
                }
            }
        };

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

                    LaunchMatMul<Device, T>::launch(ctx, a, b, dim_pair, out);
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

#define REGISTER_GPU(T)                                             \
    REGISTER_KERNEL_BUILDER(                                          \
            Name("MatMul").Device(DEVICE_GPU).TypeConstraint<T>("T"),     \
            MatMulOp<GPUDevice, T>)

    TF_CALL_float(REGISTER_CPU);
    TF_CALL_double(REGISTER_CPU);
    TF_CALL_int32(REGISTER_CPU);

    TF_CALL_float(REGISTER_GPU);
    TF_CALL_double(REGISTER_GPU);
    TF_CALL_int32(REGISTER_GPU);

}// namespace dlxnet
