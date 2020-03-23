#include "dlxnet/cc/framework/scope.h"


namespace dlxnet{


    /*static*/ Scope Scope::NewRootScope(){
        Graph* graph = new Graph(OpRegistry::Global());
        ShapeRefiner* refiner = new ShapeRefiner(graph->op_registry());
        return Scope(graph, new Status, refiner, /* disable_shape_inference */ false);
    }

    Scope::Scope(Graph* graph, Status* status, ShapeRefiner* refiner,
                    bool disable_shape_inference)
        :graph_(graph), status_(status), refiner_(refiner){}

    Scope::Scope(Scope& scope, const string& name, const string& op_name){
    }

    // op scope
    Scope Scope::WithOpName(const string& op_name) const {
        return Scope(*this, Tags::OpName(), name_, op_name);
    }

    template<typename ...Args >
    Scope Scope::WithOpName(Args... fragments) const {
        return WithOpName(absl::StrCat(fragments...));
    }
    Status Scope::ToGraphDef(GraphDef* graph){
        if (!ok()) {
            return *impl()->status_;
        }
        graph()->ToGraphDef(gdef);
        return Status::OK();
    }
}
