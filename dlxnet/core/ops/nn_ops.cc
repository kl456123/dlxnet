#include "dlxnet/core/framework/op.h"
#include "dlxnet/core/framework/shape_inference.h"
#include "dlxnet/core/framework/common_shape_fns.h"
#include "dlxnet/core/util/tensor_format.h"


namespace dlxnet{
    using shape_inference::InferenceContext;
    using shape_inference::ShapeHandle;
    // define nn ops like conv, max_pool here

    REGISTER_OP("Conv2D")
      .Input("input: T")
      .Input("filter: T")
      .Output("output: T")
      .Attr("T: {half, bfloat16, float, double, int32}")
      .Attr("strides: list(int)")
      .Attr("use_cudnn_on_gpu: bool")
      .Attr(GetConvnetDataFormatAttrString())
      .Attr("dilations: list(int)")
      .SetShapeFn(shape_inference::Conv2DShapeWithExplicitPadding);
}
