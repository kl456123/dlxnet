#ifndef DLXNET_CC_FRAMEWORK_SCOPE_H_
#define DLXNET_CC_FRAMEWORK_SCOPE_H_
#include "dlxnet/core/common_runtime/shape_refiner.h"
#include "dlxnet/core/framework/graph.pb.h"
#include "dlxnet/core/graph/graph.h"


namespace dlxnet{
    // used to construct the whole graph
    // call ToGraphDef to serialize computation graph
    class Scope{
        public:
            template<typename ...Args>
            Scope WithOpName(Args...);
            Status ToGraphDef(GraphDef* graph);

            static NewRootScope();

        private:
            ShapeRefiner* refiner_;// do shape inference
            Status* status_;// internal errors
            Graph* graph_;
    };
}

#endif

