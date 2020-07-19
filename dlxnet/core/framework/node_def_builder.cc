#include "dlxnet/core/framework/node_def_builder.h"


namespace dlxnet{
    NodeDefBuilder::NodeOut::NodeOut(StringPiece n, int i, DataType dt)
        : node(n), index(i), data_type(dt) {}

    NodeDefBuilder::NodeOut::NodeOut() {
        // uninitialized, call Reset() before use.
    }

    void NodeDefBuilder::NodeOut::Reset(StringPiece n, int i, DataType dt) {
        node = string(n);
        index = i;
        data_type = dt;
    }

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

    NodeDefBuilder& NodeDefBuilder::Input(const NodeOut& src) {
        Input(src.node, src.index, src.data_type);
        return *this;
    }

    // For inputs that take a list of tensors.
    // NodeDefBuilder& NodeDefBuilder::Input(gtl::ArraySlice<NodeOut> src_list) {
        // const OpDef::ArgDef* arg = NextArgDef();
        // if (arg != nullptr) ListInput(arg, src_list);
        // return *this;
    // }

    void NodeDefBuilder::Initialize(){
        inputs_specified_=0;
        node_def_.set_op(op_def_->name());
    }

    NodeDefBuilder::NodeDefBuilder(StringPiece name, const OpDef* op_def)
        :op_def_(op_def){
            node_def_.set_name(string(name));
            Initialize();
        }

    const OpDef::ArgDef* NodeDefBuilder::NextArgDef(){
        // next util stop
        if(!NextArgAvailable()){
            return nullptr;
        }
        return &op_def_->input_arg(inputs_specified_++);
    }

    bool NodeDefBuilder::NextArgAvailable(){
        if(op_def_==nullptr)return false;
        if(inputs_specified_>=op_def_->input_arg_size()){
            // out of range in input
            errors_.push_back(strings::StrCat("More Input() calls than the ",
                        op_def_->input_arg_size(),
                        " input_args"));
            return false;
        }
        return true;
    }

    NodeDefBuilder& NodeDefBuilder::Input(StringPiece src_node, int src_index,
            DataType dt){
        // move pointer to the next pos
        const OpDef::ArgDef* arg = NextArgDef();
        if(arg!=nullptr) SingleInput(arg, src_node, src_index, dt);
        return *this;
    }

    void NodeDefBuilder::AddInput(StringPiece src_node, int src_index) {
        if (src_node.empty()) {
            errors_.push_back("Empty input node name");
        }  else if (src_index > 0) {
            node_def_.add_input(strings::StrCat(src_node, ":", src_index));
        } else {
            node_def_.add_input(string(src_node));
        }
    }

    void NodeDefBuilder::SingleInput(const OpDef::ArgDef* input_arg, StringPiece src_node,
            int src_index, DataType dt){
        // add to node_def first
        AddInput(src_node, src_index);
        // then check it type
        if (input_arg->type() != DT_INVALID) {
            const DataType expected = MaybeAddRef(input_arg, input_arg->type());
            VerifyInputType(input_arg, expected, dt);
        } else {
            // must be reference type if datatype is invalid
            VerifyInputRef(input_arg, dt);
            Attr(input_arg->type_attr(), BaseType(dt));
        }
    }

    // verify ref and type
    void NodeDefBuilder::VerifyInputType(const OpDef::ArgDef* input_arg,
            DataType expected, DataType dt) {
        if (!TypesCompatible(expected, dt)) {
            errors_.push_back(strings::StrCat("Input '", input_arg->name(), "' passed ",
                        DataTypeString(dt), " expected ",
                        DataTypeString(expected)));
        }
    }

    void NodeDefBuilder::VerifyInputRef(const OpDef::ArgDef* input_arg,
            DataType dt) {
        if (input_arg->is_ref() && !IsRefType(dt)) {
            errors_.push_back(strings::StrCat("Input '", input_arg->name(), "' passed ",
                        DataTypeString(dt),
                        " expected ref type"));
        }
    }

    Status NodeDefBuilder::Finalize(NodeDef* node_def, bool consume){
        const std::vector<string>* errors_ptr = &errors_;
        if(op_def_!=nullptr && inputs_specified_<op_def_->input_arg_size()){
            // no enough inputs specified
            errors_.push_back(
                    strings::StrCat(inputs_specified_, " inputs specified of ",
                        op_def_->input_arg_size(), " inputs in Op"));
        }

        // return error meesage
        if(!errors_ptr->empty()){
            if (errors_ptr->size() == 1) {
                if (op_def_ == nullptr) {
                    return errors::InvalidArgument((*errors_ptr)[0],
                            " while building NodeDef '",
                            node_def_.name(), "'");
                }
                return errors::InvalidArgument(
                        (*errors_ptr)[0], " while building NodeDef '", node_def_.name(),
                        "' using ", SummarizeOpDef(*op_def_));
            } else {
                return errors::InvalidArgument(
                        errors_ptr->size(), " errors while building NodeDef '",
                        node_def_.name(), "' using ", SummarizeOpDef(*op_def_), ":\n",
                        absl::StrJoin(*errors_ptr, "\n"));
            }
        }else{
            // no error
            NodeDef node_def_backup;
            if (node_def == nullptr) node_def = &node_def_backup;
            if(consume){
                *node_def = std::move(node_def_);
            }else{
                *node_def = node_def_;
            }
            // add default value
            AddDefaultsToNodeDef(*op_def_, node_def);
            return Status::OK();
        }
    }

    NodeDefBuilder& NodeDefBuilder::Device(StringPiece device_spec){
        node_def_.set_device(string(device_spec));
        return *this;
    }

    // all attribution functions
    // for each kind of attr, they should be converted to AttrValue

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

    ATTR(StringPiece)
        ATTR(const char*)
        ATTR(int32)
        ATTR(int64)
        ATTR(float)
        ATTR(double)
        ATTR(bool)
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
