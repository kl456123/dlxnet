#include "dlxnet/core/framework/common_shape_fns.h"


namespace dlxnet{
    namespace shape_inference{
        Status UnknownShape(shape_inference::InferenceContext* c) {
            for (int i = 0; i < c->num_outputs(); ++i) {
                c->set_output(i, c->UnknownShape());
            }
            return Status::OK();
        }

        namespace {
            // helper functions
            Status Conv2DShapeImpl(shape_inference::InferenceContext* c){
                // get attr from node
                // here we just get filter shape and data format

                ShapeHandle output_shape;
                // populate output shape then set it
                c->set_output(0, output_shape);
            }
        }// namespace

        Status Conv2DShapeWithExplicitPadding(shape_inference::InferenceContext* c){
            Conv2DShapeImpl(c);
        }


        Status MatMulShape(shape_inference::InferenceContext* c){
            return Status::OK();
        }
    }// shape_inference
}// namespace dlxnet
