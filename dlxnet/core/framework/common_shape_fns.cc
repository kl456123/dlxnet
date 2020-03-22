#include "dlxnet/core/framework/common_shape_fns.h"


namespace dlxnet{
    namespace shape_inference{
        Status UnknownShape(shape_inference::InferenceContext* c) {
            for (int i = 0; i < c->num_outputs(); ++i) {
                // manage it manually
                // Shape is private in current scope
                // c->set_output(i, new Shape());
            }
            return Status::OK();
        }
    }// shape_inference
}// namespace dlxnet
