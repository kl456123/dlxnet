#ifndef DLXNET_CORE_COMMON_RUNTIME_SHAPE_REFINER_H_
#define DLXNET_CORE_COMMON_RUNTIME_SHAPE_REFINER_H_
#include <unordered_map>

#include "dlxnet/core/platform/macros.h"
#include "dlxnet/core/platform/types.h"
#include "dlxnet/core/lib/status.h"
#include "dlxnet/core/framework/shape_inference.h"
#include "dlxnet/core/graph/graph.h"
#include "dlxnet/core/framework/op.h"
#include "dlxnet/core/graph/graph.h"


namespace dlxnet{
    using shape_inference::InferenceContext;
    using shape_inference::ShapeHandle;

    class ExtendedInferenceContext{
        public:
            ExtendedInferenceContext(std::unique_ptr<shape_inference::InferenceContext>ic, Node* node)
                :inference_context_(std::move(ic)), op_(node->name()){
                    input_types_.reserve(node->num_inputs());
                    for (int i = 0; i < node->num_inputs(); i++) {
                        input_types_.push_back(node->input_type(i));
                    }
                    output_types_.reserve(node->num_outputs());
                    for (int i = 0; i < node->num_outputs(); i++) {
                        output_types_.push_back(node->output_type(i));
                    }
                }
            // accessor
            DataType input_type(int64 idx) const { return input_types_[idx]; }
            DataType output_type(int64 idx) const { return output_types_[idx]; }

            shape_inference::InferenceContext* get_context() {
                return inference_context_.get();
            }

        private:
            std::unique_ptr<shape_inference::InferenceContext> inference_context_;
            std::string op_;
            std::vector<DataType> input_types_;
            std::vector<DataType> output_types_;

            TF_DISALLOW_COPY_AND_ASSIGN(ExtendedInferenceContext);

    };
    class ShapeRefiner{
        public:
            ShapeRefiner(const OpRegistryInterface* ops);
            ~ShapeRefiner();

            // Performs validation of 'node' and runs 'node's shape function,
            // storing its shape outputs.
            //
            // All inputs of 'node' must be added to ShapeRefiner prior to
            // adding 'node'.
            //
            // Returns an error if:
            //  - the shape function for 'node' was not registered.
            //  - 'node' was added before its inputs.
            //  - The shape inference function returns an error.
            Status AddNode(const Node* node);

            Status SetShape(const Node* node, int output_port, ShapeHandle shape);

            Status RunShapeFn(const Node* node, const OpRegistrationData* op_reg_data,
                    ExtendedInferenceContext* ec);

            // Returns the InferenceContext for 'node', if present.
            shape_inference::InferenceContext* GetContext(const Node* node) const {
                auto it = node_to_context_.find(node);
                if (it == node_to_context_.end()) {
                    return nullptr;
                }
                return it->second->get_context();
            }
            void set_require_shape_inference_fns(bool require_shape_inference_fns) {
                require_shape_inference_fns_ = require_shape_inference_fns;
            }
        private:
            const OpRegistryInterface* const ops_registry_;
            // Stores a map from a node to its ExtendedInferenceContext.
            std::unordered_map<const Node*, std::unique_ptr<ExtendedInferenceContext>>
                node_to_context_;
            bool require_shape_inference_fns_;
            TF_DISALLOW_COPY_AND_ASSIGN(ShapeRefiner);
    };
}


#endif
