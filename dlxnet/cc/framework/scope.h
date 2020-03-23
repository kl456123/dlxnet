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
            Scope(Scope& other);
            ~Scope();
            template<typename ...Args>
                Scope WithOpName(Args...)const;
            Status ToGraphDef(GraphDef* graph);

            static Scope NewRootScope();

        private:
            Scope(Graph* graph, Status* status, ShapeRefiner* refiner,
                    bool disable_shape_inference);
            ShapeRefiner* refiner_;// do shape inference
            Status* status_;// internal errors
            Graph* graph_;
    };
}

#endif

