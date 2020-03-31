#ifndef DLXNET_CORE_KERNELS_MATMUL_H_
#define DLXNET_CORE_KERNELS_MATMUL_H_
#include "dlxnet/core/framework/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace dlxnet{
    namespace functor{
        template<typename T>
            struct MatMulTypes{
                typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned>
                    out_type;
                typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
                        Eigen::Aligned>
                            in_type;
            };
        template <typename Device, typename In0, typename In1, typename Out,
                 typename DimPair>
                     void MatMul(const Device& d, Out out, In0 in0, In1 in1,
                             const DimPair& dim_pair) {
                         out.device(d) = in0.contract(in1, dim_pair);
                     }

        template<typename Device, typename T>
            struct MatMulFunctor{
                // Computes on device "d": out = in0 * in1, where * is matrix
                // multiplication.
                void operator()(
                        const Device& d, typename MatMulTypes<T>::out_type out,
                        typename MatMulTypes<T>::in_type in0,
                        typename MatMulTypes<T>::in_type in1,
                        const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair);
            };
    } //namespace functor

}//namespace dlxnet


#endif
