#ifndef DLXNET_CORE_COMMON_RUNTIME_SHAPE_REFINER_H_
#define DLXNET_CORE_COMMON_RUNTIME_SHAPE_REFINER_H_
#include "dlxnet/core/platform/macros.h"
#include "dlxnet/core/lib/status.h"


namespace dlxnet{
    class ShapeRefiner{
        public:
            ShapeRefiner();
            ~ShapeRefiner();

            Status SetShape(const Node* node, int output_port, shape);
            TF_DISALLOW_COPY_AND_ASSIGN(ShapeRefiner);
    };
}


#endif
