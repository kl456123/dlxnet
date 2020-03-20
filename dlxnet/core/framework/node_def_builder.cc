#include "dlxnet/core/framework/node_def_builder.h"


namespace dlxnet{
    NodeDefBuilder::NodeDefBuilder(StringPiece name, StringPiece op_name,
            const OpRegistryInterface* op_registry){
        node_def_.set_name(string(name));
        const Status status = op_registry->LookUpOpDef(string(op_name), &op_def_);
        if(status.ok()){
            Initialize();
        }else{
            // push error to errors_
            errors_.push_back(status.error_message());
        }
    }

    void NodeDefBuilder::Initialize(){
        inputs_specified_=0;
        node_def_.set_op(op_def_->name());
    }

    NodeDefBuilder& NodeDefBuilder::Input(StringPiece src_node, int src_index,
            DataType dt){
    }

    Status NodeDefBuilder::Finalize(NodeDef* node_def){
        if(op_def_!=nullptr && inputs_specified_<op_def_->input_arg_size()){
            // no enough inputs specified
            errors_.push_back(
                    strings::StrCat(inputs_specified_, " inputs specified of ",
                        op_def_->input_arg_size(), " inputs in Op"));
        }
        // return error meesage
        if(!errors_.empty()){
            // one error
            if(errors_.size()==1){
                return errors::InvalidArgument(errors_[0],
                        " while building NodeDef '",
                        node_def_.name(), "'");
            }else{
                // summary all
            }
        }else{
            // no error
            *node_def = node_def_;
            // add default value
            AddDefaultsToNodeDef(*op_def_, node_def);
            return Status::OK();
        }
    }
}
