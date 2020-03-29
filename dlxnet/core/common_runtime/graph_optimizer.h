#ifndef DLXNET_CORE_COMMON_RUNTIME_GRAPH_OPTIMIZER_H_
#define DLXNET_CORE_COMMON_RUNTIME_GRAPH_OPTIMIZER_H_
#include <memory>

#include "dlxnet/core/common_runtime/device.h"
#include "dlxnet/core/platform/env.h"
#include "dlxnet/core/graph/graph.h"
#include "dlxnet/core/protobuf/config.pb.h"

namespace dlxnet{
    class GraphOptimizer{
        public:
            explicit GraphOptimizer(const OptimizerOptions& opts);
            ~GraphOptimizer();

            // Applies optimization passes specified in 'opts' to 'graph'.
            // Maybe replace *graph with a new graph object.  'device' is device
            // on which the 'graph' will execute. It's passed to the optimizers
            // so that they can respect constraints if any, that should be
            // respected.
            void Optimize(Env* env, const Device* device,
                    std::unique_ptr<Graph>* graph,
                    const OptimizerOptions& graph_optimizer_options);
        private:
            OptimizerOptions opts_;
            TF_DISALLOW_COPY_AND_ASSIGN(GraphOptimizer);
    };
}


#endif
