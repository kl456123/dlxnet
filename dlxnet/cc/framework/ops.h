#ifndef DLXNET_CC_FRAMEWORK_OPS_H_
#define DLXNET_CC_FRAMEWORK_OPS_H_
#include "dlxnet/core/graph/graph.h"

namespace dlxnet{
    // just wrap node and edge
    class Output;
    class Input;
    class Operation;

    class Operation{
        public:
            Operation(Node* node);
    };

    class Input{
    };

    class Output{
    };

}


#endif
