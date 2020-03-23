#include "dlxnet/core/framework/op.h"
#include "dlxnet/core/framework/shape_inference.h"

namespace dlxnet{

    using shape_inference::InferenceContext;
    using shape_inference::ShapeHandle;
    using shape_inference::DimensionHandle;

    // define many perimitive ops here
    // --------------------------------------------------------------------------
    REGISTER_OP("Placeholder")
        .Output("output: dtype")
        .Attr("dtype: type")
        // cannot parse default value due to lack of pb_txt
        // .Attr("shape: shape = { unknown_rank: true }")
        .SetShapeFn([](InferenceContext* c){
                ShapeHandle out;
                c->set_output(0, out);
                return Status::OK();
                });

    // --------------------------------------------------------------------------
  REGISTER_OP("Const")
      .Output("output: dtype")
      .Attr("value: tensor")
      .Attr("dtype: type")
      .SetShapeFn([](InferenceContext* c) {
        const TensorProto* proto = nullptr;
        TF_RETURN_IF_ERROR(c->GetAttr("value", &proto));
        TF_RETURN_IF_ERROR(TensorShape::IsValidShape(proto->tensor_shape()));
        TensorShape shape(proto->tensor_shape());
        std::vector<DimensionHandle> dims;
        dims.reserve(shape.dims());
        for (int i = 0; i < shape.dims(); ++i) {
          dims.push_back(c->MakeDim(shape.dim_size(i)));
        }
        c->set_output(0, c->MakeShape(dims));
        return Status::OK();
      });
} // namespace dlxnet
