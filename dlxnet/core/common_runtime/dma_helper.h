#ifndef DLXNET_CORE_COMMON_RUNTIME_DMA_HELPER_H_
#define DLXNET_CORE_COMMON_RUNTIME_DMA_HELPER_H_
#include "dlxnet/core/framework/tensor.h"

namespace dlxnet{
    class DMAHelper {
        public:
            static bool CanUseDMA(const Tensor* t) { return t->CanUseDMA(); }
            static const void* base(const Tensor* t) { return t->base<const void>(); }
            static void* base(Tensor* t) { return t->base<void>(); }
            static TensorBuffer* buffer(Tensor* t) { return t->buf_; }
            static const TensorBuffer* buffer(const Tensor* t) { return t->buf_; }
            static void UnsafeSetShape(Tensor* t, const TensorShape& s) {
                t->set_shape(s);
            }
    };
} // namespace dlxnet


#endif
