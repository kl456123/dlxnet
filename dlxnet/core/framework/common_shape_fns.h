#ifndef DLXNET_CORE_FRAMEWORK_COMMON_SHAPE_FNS_H_
#define DLXNET_CORE_FRAMEWORK_COMMON_SHAPE_FNS_H_
#include "dlxnet/core/framework/shape_inference.h"


namespace dlxnet{
    namespace shape_inference{
        Status UnknownShape(shape_inference::InferenceContext* c);
    }
}


#endif
