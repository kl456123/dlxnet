#include "dlxnet/core/framework/op.h"
#include "dlxnet/core/framework/attr_value.pb.h"
#include "dlxnet/core/framework/common_shape_fns.h"
#include "dlxnet/core/framework/shape_inference.h"
#include "dlxnet/core/framework/tensor_shape.pb.h"
#include "dlxnet/core/lib/core/errors.h"

namespace dlxnet{
    REGISTER_SYSTEM_OP("_Arg")
        .Output("output: T")
        .Attr("T: type")
        .Attr("index: int >= 0")
        .SetShapeFn([](shape_inference::InferenceContext* context) {
                const AttrValue* shape_attr = context->attrs().Find("_output_shapes");
                if (shape_attr && shape_attr->has_list()) {
                if (shape_attr->list().shape().empty()) {
                return errors::InvalidArgument(
                        "Invalid \"_output_shapes\" attribute value for _Arg node: ",
                        shape_attr->DebugString());
                }
                const TensorShapeProto& shape_proto = shape_attr->list().shape(0);
                shape_inference::ShapeHandle shape_handle;
                TF_RETURN_IF_ERROR(
                        context->MakeShapeFromShapeProto(shape_proto, &shape_handle));
                context->set_output(0, shape_handle);
                } else {
                context->set_output(0, context->UnknownShape());
                }
                return Status::OK();
                })
    .Doc(R"doc(
A graph node which represents an argument to a function.

output: The argument.
index: This argument is the index-th argument of the function.

Attributes for shape inference:
1. _output_shapes: this attribute can be set on an _Arg node producing
   non-resource output(s). If set, its value should contain a list of
   TensorShapeProto describing the shape(s) of the tensor(s) this _Arg node will
   produce. If set, _Arg node's shape inference function will use it as the
   node's output shapes.
2. _handle_dtypes and _handle_shapes: these attributes can be set on an _Arg
   node producing resource output(s). If set, value of _handle_dtypes should
   contain the dtype(s) of the resource(s) and value of _handle_shapes should
   contain the shape(s) of the resource(s). If both attributes are set, _Arg
   node's shape inference function will use their values as the node's output
   type(s) and shape(s).
)doc");

    REGISTER_SYSTEM_OP("_Retval")
    .Input("input: T")
    .Attr("T: type")
    .Attr("index: int >= 0")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* context) {
            return Status::OK();
            })
    .Doc(R"doc(
A graph node which represents a return value of a function.

input: The return value.
index: This return value is the index-th return value of the function.
)doc");
}
