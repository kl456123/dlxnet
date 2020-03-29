#include "dlxnet/core/common_runtime/graph_optimizer.h"


namespace dlxnet{
    GraphOptimizer::GraphOptimizer(const OptimizerOptions& opts){
    }
    GraphOptimizer::~GraphOptimizer(){}
    void GraphOptimizer::Optimize(Env* env, const Device* device,
            std::unique_ptr<Graph>* graph,
            const OptimizerOptions& graph_optimizer_options){
    }
} //namespace dlxnet
