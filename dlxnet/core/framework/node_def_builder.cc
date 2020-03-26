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

    NodeDefBuilder::NodeDefBuilder(StringPiece name, const OpDef* op_def)
        :op_def_(op_def){
            node_def_.set_name(string(name));
            Initialize();
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

    NodeDefBuilder& NodeDefBuilder::Device(StringPiece device_spec){
        node_def_.set_device(string(device_spec));
    }

    // all attribution functions
    // NodeDefBuilder& NodeDefBuilder::Attr(StringPiece name, const TensorProto& value){
    // return *this;
    // }
    // NodeDefBuilder& NodeDefBuilder::Attr(StringPiece name, const Tensor& value){
    // return *this;
    // }

    // NodeDefBuilder& NodeDefBuilder::Attr(StringPiece name, int32 value){
    // return *this;
    // }

    bool NodeDefBuilder::AttrValueAlreadyPresent(StringPiece name,
            const AttrValue& value) {
        if (const AttrValue* found = AttrSlice(node_def_).Find(name)) {
            return true;
        }
        return false;
    }

    NodeDefBuilder& NodeDefBuilder::Attr(StringPiece name, const AttrValue& value) {
        if (!AttrValueAlreadyPresent(name, value)) {
            AddNodeAttr(name, value, &node_def_);
        }
        return *this;
    }

    NodeDefBuilder& NodeDefBuilder::Attr(StringPiece name, AttrValue&& value) {
        if (!AttrValueAlreadyPresent(name, value)) {
            AddNodeAttr(name, std::move(value), &node_def_);
        }
        return *this;
    }


#define ATTR(T)                                                         \
        NodeDefBuilder& NodeDefBuilder::Attr(StringPiece name, T value) {   \
            AttrValue attr_value;                                           \
            SetAttrValue(value, &attr_value);                               \
            return Attr(name, attr_value);                                  \
        }

        // ATTR(StringPiece)
        // ATTR(const char*)
        ATTR(int32)
            // ATTR(int64)
            ATTR(float)
            // ATTR(double)
            // ATTR(bool)
            ATTR(DataType)
            ATTR(const Tensor&)
            ATTR(const TensorProto&)
            // ATTR(const NameAttrList&)
            // ATTR(gtl::ArraySlice<StringPiece>)
            // ATTR(gtl::ArraySlice<const char*>)
            // ATTR(gtl::ArraySlice<string>)
            // ATTR(gtl::ArraySlice<int32>)
            // ATTR(gtl::ArraySlice<int64>)
            // ATTR(gtl::ArraySlice<float>)
            // ATTR(gtl::ArraySlice<bool>)
            // ATTR(const std::vector<bool>&)
            // ATTR(gtl::ArraySlice<DataType>)
            // ATTR(gtl::ArraySlice<TensorShape>)
            // ATTR(gtl::ArraySlice<PartialTensorShape>)
            // ATTR(gtl::ArraySlice<TensorShapeProto>)
            // ATTR(gtl::ArraySlice<Tensor>)
            // ATTR(gtl::ArraySlice<NameAttrList>)
#undef ATTR


    }
