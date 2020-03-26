#ifndef DLXNET_CC_FRAMEWORK_OPS_H_
#define DLXNET_CC_FRAMEWORK_OPS_H_
#include "dlxnet/core/graph/graph.h"
#include "dlxnet/core/framework/types.pb.h"
#include "dlxnet/core/framework/tensor.h"
#include "dlxnet/core/framework/tensor_shape.h"

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
                // single number or string
                template<typename T, typename = typename std::enable_if<
                    std::is_convertible<T, string>::value ||
                    std::is_arithmetic<T>::value>::type>
                    Initializer(const T& value){
                        // create empty tensor using type and shape
                        Tensor t(DataTypeToEnum<T>::v(), TensorShape());
                        // assign value
                        t.flat<T>()[0] = value;
                        tensor = t ;
                    }

                // list of single number of string
                /// Construct from a initializer list of scalars (a one-dimensional tensor).
                template <typename T, typename = typename std::enable_if<
                    std::is_arithmetic<T>::value ||
                    std::is_convertible<T, string>::value>::type>
                    Initializer(
                            const std::initializer_list<T>& v){
                        Tensor t(DataTypeToEnum<T>::v(),
                                TensorShape{static_cast<int>(v.size())});
                        std::copy_n(v.begin(), v.size(), t.flat<T>().data());
                        tensor = t;
                    }

                Initializer(const Tensor& t) : tensor(t) {}  // NOLINT(runtime/explicit)

                /// Construct a multi-dimensional tensor from a nested initializer
                /// list. Note that C++ syntax allows nesting of arbitrarily typed
                /// initializer lists, so such invalid initializers cannot be disallowed at
                /// compile time. This function performs checks to make sure that the nested
                /// initializer list is indeed a valid multi-dimensional tensor.
                Initializer(const std::initializer_list<Initializer>& v);
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
