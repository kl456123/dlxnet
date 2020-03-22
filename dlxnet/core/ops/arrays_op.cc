#include "dlxnet/core/framework/op.h"
#include "dlxnet/core/framework/shape_inference.h"

namespace dlxnet{

    using shape_inference::InferenceContext;
    using shape_inference::ShapeHandle;

    // define many ops here
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
} // namespace dlxnet
