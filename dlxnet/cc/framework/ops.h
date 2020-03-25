#ifndef DLXNET_CC_FRAMEWORK_OPS_H_
#define DLXNET_CC_FRAMEWORK_OPS_H_
#include "dlxnet/core/graph/graph.h"
#include "dlxnet/core/framework/types.pb.h"
#include "dlxnet/core/framework/tensor.h"

namespace dlxnet{
    // just wrap node and edge
    class Output;
    class Input;
    class Operation;

    class Operation{
        public:
            Operation(Node* node);
    };

    // container for scalar type(from user input) and node type(output from operation)
    class Input{
        public:
            // from scalar type or list type to tensor
            struct Initializer{
                Tensor tensor;
                Status status;
            };
            // some useful constructors, list or single input
            Input(Node* node):node_(node){}
            Input(const Tensor& t)  // NOLINT(runtime/explicit)
                : status_(Status::OK()),
                tensor_(t) {}
            // some initializers
            Input(const Initializer& init)
                :status_(init.status),
                tensor_(init.tensor){}

            // accessor
            Node* node()const{return node_;}
            int32 index() const { return index_; }
            DataType data_type() const { return data_type_; }
            Status status() const { return status_; }
            const Tensor& tensor() const { return tensor_; }
        private:
            // the two most important things are index and node
            Status status_;
            Tensor tensor_;
            int32 index_=0;
            Node* node_;
            DataType data_type_ = DT_INVALID;
    };

    class Output{
    };

}


#endif
