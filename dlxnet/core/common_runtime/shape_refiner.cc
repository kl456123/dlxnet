#include "dlxnet/core/common_runtime/shape_refiner.h"
#include "dlxnet/core/framework/common_shape_fns.h"


namespace dlxnet{
    ShapeRefiner::ShapeRefiner(const OpRegistryInterface* ops)
        :ops_registry_(ops){
        }

    ShapeRefiner::~ShapeRefiner(){
    }
    Status ShapeRefiner::AddNode(const Node* node){
        // Create the inference context for this node with the existing input shapes.
        std::unique_ptr<InferenceContext> ic(new InferenceContext(
                    node->def(), node->op_def(),
                    std::vector<ShapeHandle>(node->num_inputs())));

        // get input shape
        for (const Edge* e : node->in_edges()){
            const Node* input = e->src();
            auto it = node_to_context_.find(input);
            if (it == node_to_context_.end()){
                return errors::Internal("Cannot find input shape in index slot: ",
                        e->dst_input()," for node. ");
            }

            InferenceContext* input_ic = it->second->get_context();
            // set input shape in x-th index of input slot of the node
            ic->SetInput(e->dst_input(), input_ic->output(e->src_output()));
        }

        // prepare to run shape inference
        // Get the shape function for this node
        const OpRegistrationData* op_reg_data;
        TF_RETURN_IF_ERROR(ops_registry_->LookUp(node->type_string(), &op_reg_data));
        // may be some node dont need to do shape inference
        if (op_reg_data->shape_inference_fn == nullptr &&
                require_shape_inference_fns_) {
            return errors::InvalidArgument(
                    "No shape inference function exists for op '", node->type_string(),
                    "', did you forget to define it?");
        }
        // get extend context to cache and run.
        std::unique_ptr<ExtendedInferenceContext> ec(
                new ExtendedInferenceContext(std::move(ic), node));

        // just run it
        TF_RETURN_IF_ERROR(RunShapeFn(node, op_reg_data, ec.get()));

        // cache it in the map
        node_to_context_[node].swap(ec);

        return Status::OK();
    }
    Status ShapeRefiner::RunShapeFn(const Node* node, const OpRegistrationData* op_reg_data,
            ExtendedInferenceContext* ec){
        auto* c = ec->get_context();
        auto run_inference_lambda = [&](){
            if(op_reg_data->shape_inference_fn){
                TF_RETURN_IF_ERROR(c->Run(op_reg_data->shape_inference_fn));
            }else{
                TF_RETURN_IF_ERROR(c->Run(shape_inference::UnknownShape));
            }
            return Status::OK();
        };
        TF_RETURN_IF_ERROR(run_inference_lambda());

        //(TODO) may be need to change shape dynamically
    }

}
