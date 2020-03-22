#include "dlxnet/core/framework/shape_inference.h"
#include "dlxnet/core/framework/node_def_util.h"


namespace dlxnet{
    namespace shape_inference{
        // do inference just for single node
        InferenceContext::InferenceContext(const NodeDef& node_def,
                const OpDef& op_def)
            :node_def_(node_def){
                // init inputs and outputs
                PreInputInit(op_def);
                // inputs_.reserve();
            }

        void InferenceContext::PreInputInit(const OpDef& op_def){
            construction_status_ =
                NameRangesForNode(node_def_, op_def, &input_name_map_, &output_name_map_);
            if (!construction_status_.ok()) return;

            int num_outputs = 0;
            for (const auto& e : output_name_map_) {
                num_outputs = std::max(num_outputs, e.second.second);
            }
            outputs_.assign(num_outputs, nullptr);
        }


        InferenceContext::~InferenceContext(){
        }
        Status InferenceContext::Run(
                const std::function<Status(shape_inference::InferenceContext* c)>& fn){
            Status s = fn(this);
            if(!s.ok()){
                // do something more(like attach information) for debugging
            }
            return s;
        }

        string InferenceContext::DebugString() const{
        }
    }
}
