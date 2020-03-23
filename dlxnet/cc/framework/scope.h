#ifndef DLXNET_CC_FRAMEWORK_SCOPE_H_
#define DLXNET_CC_FRAMEWORK_SCOPE_H_
#include "dlxnet/core/common_runtime/shape_refiner.h"
#include "dlxnet/core/framework/graph.pb.h"
#include "dlxnet/core/graph/graph.h"


namespace dlxnet{
    // used to construct the whole graph
    // call ToGraphDef to serialize computation graph
    struct Tags{
        enum class ScopeName;
        enum class OpName;
    };
    class Scope{
        public:
            // Scope(Scope& other);
            ~Scope();
            template<typename ...Args>
                Scope WithOpName(Args...)const;
            Scope WithOpName(const string& op_name) const;

            Status ToGraphDef(GraphDef* graph);

            static Scope NewRootScope();

            // accessor
            Graph* graph(){return graph_;}
            bool ok(){return status_.ok();}

        private:
            Scope(Graph* graph, Status* status, ShapeRefiner* refiner,
                    bool disable_shape_inference);
            ShapeRefiner* refiner_;// do shape inference
            Status* status_;// internal errors
            Graph* graph_;
            string name_;
    };
}

#endif

