#include "dlxnet/core/framework/op.h"
#include "dlxnet/core/framework/shape_inference.h"
#include "dlxnet/core/framework/common_shape_fns.h"

namespace dlxnet{
    using shape_inference::InferenceContext;
    using shape_inference::ShapeHandle;
    using shape_inference::DimensionHandle;

    // --------------------------------------------------------------------------
    REGISTER_OP("MatMul")
        .Input("a: T")
        .Input("b: T")
        .Output("product: T")
        .Attr("transpose_a: bool")
        .Attr("transpose_b: bool")
        .Attr("T: {float, double, int32, int64}")
        .SetShapeFn(shape_inference::MatMulShape);


}// namespace dlxnet
