#include "dlxnet/core/framework/shape_inference.h"
#include "dlxnet/core/framework/node_def_util.h"


namespace dlxnet{
    namespace shape_inference{
        // do inference just for single node
        InferenceContext::InferenceContext(const NodeDef& node_def,
                const OpDef& op_def, const std::vector<ShapeHandle>& input_shapes)
            :node_def_(node_def){
                // init inputs and outputs
                // fill input_name_map_ and output_name_map_,
                // init outputs_ to nullptr
                PreInputInit(op_def);
                if(!construction_status_.ok())return;
                inputs_ = input_shapes;
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

        Status InferenceContext::input(StringPiece input_name, std::vector<ShapeHandle>* output) const{
            const auto result = input_name_map_.find(input_name);
            if(result==input_name_map_.end()){
                return errors::InvalidArgument("Unknown input name: ", input_name);
            }
            output->clear();
            for(int i=result->second.first;i<result->second.second;++i){
                output->push_back(inputs_[i]);
            }
            return Status::OK();
        }

        Status InferenceContext::output(StringPiece output_name,
                std::vector<ShapeHandle>* output) const {
            const auto result = output_name_map_.find(output_name);
            if (result == output_name_map_.end()) {
                return errors::InvalidArgument("Unknown output name: ", output_name);
            } else {
                output->clear();
                for (int i = result->second.first; i < result->second.second; ++i) {
                    output->push_back(outputs_[i]);
                }
            }
            return Status::OK();
        }

        Status InferenceContext::Run(
                const std::function<Status(shape_inference::InferenceContext* c)>& fn){
            Status s = fn(this);
            if(!s.ok()){
                // do something more(like attach information) for debugging
            }
            return s;
        }

        ShapeHandle InferenceContext::MakeShape(
                const std::vector<DimensionHandle>& dims) {
            all_shapes_.push_back(new Shape(dims));
            return all_shapes_.back();
        }


        string InferenceContext::DebugString() const{
        }

        Status InferenceContext::MakeShapeFromShapeProto(const TensorShapeProto& proto,
                ShapeHandle* out) {
            *out = nullptr;
            TensorShape shape(proto);
            const int num_dims = shape.dims();
            std::vector<DimensionHandle> dims(num_dims);
            for (int i = 0; i < num_dims; ++i) {
                dims[i] = MakeDim(shape.dim_size(i));
            }
            *out = MakeShape(dims);
            return Status::OK();

        }
        ShapeHandle InferenceContext::UnknownShape() {
            all_shapes_.push_back(new Shape());
            return all_shapes_.back();
        }

    }
}
