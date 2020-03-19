#include "dlxnet/core/framework/node_def_builder.h"


namespace dlxnet{
    NodeDefBuilder::NodeDefBuilder(StringPiece name, StringPiece op_name,
            const OpRegistryInterface* op_reistry){
        node_def_set_name(string(name));
        const Status status = op_registry->LookupOpDef(string(op_name));
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

    Status Finalize(NodeDef* node_def){
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
                return errors::InvalidArgument();
            }else{
                // summary all
            }
        }else{
            // no error
            // add default value
            AddDefaultToNodeDef();
            return Status::OK();
        }
    }
    NodeDefBuilder& NodeDefBuilder::Input(StringPiece src_node, int src_index,  DataType dt){
    }
}
