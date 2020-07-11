#include "dlxnet/core/framework/node_def_util.h"
#include "dlxnet/core/framework/attr_value_util.h"
#include "dlxnet/core/framework/op_def_util.h"
#include "dlxnet/core/graph/graph.h"
#include "dlxnet/core/platform/protobuf.h"


namespace dlxnet{
    namespace {

        // note that sig refers to types_vector
        Status AddArgToSig(const NodeDef& node_def,
                const OpDef::ArgDef& arg_def, DataTypeVector* sig){
            // four cases(three normal cases and one abnormal case)
            if(!arg_def.number_attr().empty()){
                // repeated times
                int32 repeats = -1;
                TF_RETURN_IF_ERROR(GetNodeAttr(node_def, arg_def.number_attr(), &repeats));
                if(repeats<0){
                    return errors::InvalidArgument("Value for number_attr() ", repeats,
                            " < 0");
                }
                // find which type to use from type_attr or type field
                if(!arg_def.type_attr().empty()){
                    DataType dtype;
                    TF_RETURN_IF_ERROR(
                            GetNodeAttr(node_def, arg_def.type_attr(), &dtype));
                    for(int i=0;i<repeats;++i){
                        sig->push_back(dtype);
                    }
                }else if(arg_def.type()!=DT_INVALID){
                    for(int i=0;i<repeats;++i){
                        sig->push_back(arg_def.type());
                    }
                }
                else{
                    return errors::InvalidArgument("Missing type or type_attr field in ",
                            arg_def.ShortDebugString());
                }
            }else if(!arg_def.type_attr().empty()){
                // single types defined in attr
                const AttrValue* attr_value;// no need to allocate new memory
                TF_RETURN_IF_ERROR(AttrSlice(node_def).Find(arg_def.type_attr(), &attr_value));
                sig->push_back(attr_value->type());
            }else if(!arg_def.type_list_attr().empty()){
                // list of types
                const AttrValue* attr_value;
                TF_RETURN_IF_ERROR(AttrSlice(node_def).Find(arg_def.type_list_attr(), &attr_value));
                for(int dtype: attr_value->list().type()){
                    sig->push_back(static_cast<DataType>(dtype));
                }
            }else if(arg_def.type()!=DT_INVALID){
                // single perimitive type
                sig->push_back(arg_def.type());
            }
            else{
                // return error
                return errors::InvalidArgument("No type fields in ",
                        arg_def.ShortDebugString());
            }

            return Status::OK();
        }
    } // namespace

    bool HasNodeAttr(const NodeDef& node_def, StringPiece attr_name) {
        return node_def.attr().find(string(attr_name)) != node_def.attr().end();
    }
    void AddDefaultsToNodeDef(const OpDef& op_def, NodeDef* node_def){
        for(const auto& attr_def: op_def.attr()){
            AttrSlice attrs(*node_def);
            if(attr_def.has_default_value()&&!attrs.Find(attr_def.name())){
                AddNodeAttr(attr_def.name(), attr_def.default_value(), node_def);
            }
        }
    }

    void AddNodeAttr(StringPiece name, const AttrValue& value, NodeDef* node_def){
        node_def->mutable_attr()->insert(AttrValueMap::value_type(string(name), value));
    }

    void AddNodeAttr(StringPiece name, AttrValue&& value, NodeDef* node_def) {
        (*node_def->mutable_attr())[string(name)] = std::move(value);
    }

    string SummarizeNode(const Node& node){
        return SummarizeNodeDef(node.def());
    }

    string SummarizeNodeDef(const NodeDef& node_def) {
        string ret = strings::StrCat(errors::FormatNodeNameForError(node_def.name()),
                " = ", node_def.op(), "[");
        strings::StrAppend(&ret, SummarizeAttrsHelper(node_def, node_def.device()));
        strings::StrAppend(&ret, "](");

        // Output inputs, including control inputs, verbatim.
        bool first = true;
        for (const string& input : node_def.input()) {
            if (!first) strings::StrAppend(&ret, ", ");
            first = false;
            strings::StrAppend(&ret, input);
        }
        strings::StrAppend(&ret, ")");
        return ret;
    }

    string SummarizeAttrs(const NodeDef& node_def) {
        return SummarizeAttrsHelper(node_def, node_def.device());
    }

    string SummarizeAttrsHelper(AttrSlice attrs, StringPiece device){
        string ret;

        std::vector<string> attr_names;
        attr_names.reserve(attrs.size());
        for(const auto&attr: attrs){
            attr_names.push_back(attr.first);
        }

        std::sort(attr_names.begin(), attr_names.end());
        bool first = true;
        for (const string& attr_name : attr_names) {
            if (!first) strings::StrAppend(&ret, ", ");
            first = false;
            strings::StrAppend(&ret, attr_name, "=",
                    SummarizeAttrValue(*attrs.Find(attr_name)));
        }
        // Consider the device to be a final attr with name "_device".
        if (!device.empty()) {
            if (!first) strings::StrAppend(&ret, ", ");
            first = false;
            strings::StrAppend(&ret, "_device=\"", device, "\"");
        }
        return ret;
    }

    // AttrValue implementation

    AttrSlice::AttrSlice(const NodeDef& node_def)
        :ndef_(&node_def), attrs_(&ndef_->attr()){}

    AttrSlice::AttrSlice() : ndef_(nullptr) {
        static const AttrValueMap* const kEmptyAttrValueMap = new AttrValueMap;
        attrs_ = kEmptyAttrValueMap;
    }

    AttrSlice::AttrSlice(const AttrValueMap* a) : ndef_(nullptr), attrs_(a) {}

    const AttrValue* AttrSlice::Find(StringPiece attr_name) const {
        // Currently, the collection used for NodeDef::attr() (google::protobuf::Map)
        // requires that the keys used for lookups have type 'const string&'. Because
        // this method takes a StringPiece, it is necessary to allocate a temporary
        // string, copy attr_name to it, and then use that temporary string for the
        // lookup. This causes an excessive number of short-lived allocations, and for
        // large graphs, this can be a significant cost.
        //
        // Because most nodes have a small number of attributes, a simple linear scan
        // is generally more efficient than a hashed lookup.  If google::protobuf::Map
        // changes so that it supports efficient lookups using StringPiece instead of
        // const string&, then this code could be changed to use attrs_->find() again.

        for (const auto& attr : *attrs_) {
            if (attr.first == attr_name) {
                return &attr.second;
            }
        }
        return nullptr;
    }

    Status AttrSlice::Find(StringPiece attr_name,
            const AttrValue** attr_value) const {
        *attr_value = Find(attr_name);
        if (*attr_value != nullptr) {
            return Status::OK();
        }
        Status s = errors::NotFound("No attr named '", attr_name, "' in NodeDef:");
        return s;
    }

    string AttrSlice::DebugString() const {
        std::vector<string> attr_key_vals;
        attr_key_vals.reserve(attrs_->size());
        for (const auto& it : *this) {
            const string& name = it.first;
            const AttrValue& attr_value = it.second;
            attr_key_vals.push_back(
                    absl::StrCat(name, "=", SummarizeAttrValue(attr_value)));
        }
        return absl::StrJoin(attr_key_vals, ", ");
    }

    Status ValidateNodeDef(const NodeDef& node_def, const OpDef& op_def){
        if (node_def.op() != op_def.name()) {
            return errors::InvalidArgument(
                    "NodeDef op '", node_def.op(), "' does not match ",
                    SummarizeOpDef(op_def), "; NodeDef: ", "...");
        }
    }

    Status InOutTypesForNode(const NodeDef& node_def, const OpDef& op_def,
            DataTypeVector* inputs, DataTypeVector* outputs) {
        TF_RETURN_IF_ERROR(InputTypesForNode(node_def, op_def, inputs));
        return OutputTypesForNode(node_def, op_def, outputs);
    }

    Status InputTypesForNode(const NodeDef& node_def, const OpDef& op_def,
            DataTypeVector* inputs){
        for(const auto& arg: op_def.input_arg()){
            // get signature from arg
            TF_RETURN_IF_ERROR(AddArgToSig(node_def, arg, inputs));
        }
        return Status::OK();
    }

    Status OutputTypesForNode(const NodeDef& node_def, const OpDef& op_def,
            DataTypeVector* outputs){
        for(const auto& arg: op_def.output_arg()){
            // get signature from arg
            TF_RETURN_IF_ERROR(AddArgToSig(node_def, arg, outputs));
        }
        return Status::OK();
    }

    namespace {  // Helpers for NameRangesForNode()
        Status ComputeArgRange(const AttrSlice& attrs, const OpDef::ArgDef& arg_def,
                const OpDef& op_def, int* num){
            if (!arg_def.number_attr().empty()) {
                // Same type repeated "num" times.
                return GetNodeAttr(attrs, arg_def.number_attr(), num);
            }else if(!arg_def.type_list_attr().empty()){
                const AttrValue* attr_value;
                TF_RETURN_IF_ERROR(attrs.Find(arg_def.type_list_attr(), &attr_value));
                *num = attr_value->list().type_size();
            }else if(!arg_def.type_attr().empty() || arg_def.type() != DT_INVALID){
                *num = 1;
            }else{
                return errors::InvalidArgument(
                        "Argument '", arg_def.name(),
                        "' incorrectly specified in op definition: ", SummarizeOpDef(op_def));
            }
            return Status::OK();
        }
        Status NameRangesHelper(const AttrSlice& attrs,
                const protobuf::RepeatedPtrField<OpDef::ArgDef>& args,
                const OpDef& op_def, NameRangeMap* result) {
            int start = 0;
            int num;
            for (const auto& arg : args) {
                TF_RETURN_IF_ERROR(ComputeArgRange(attrs, arg, op_def, &num));
                (*result)[arg.name()] = std::make_pair(start, start + num);
                start += num;
            }
            return Status::OK();
        }
    }// namespace

    Status NameRangesForNode(const AttrSlice& attrs, const OpDef& op_def,
            NameRangeMap* inputs, NameRangeMap* outputs) {
        if (inputs != nullptr) {
            TF_RETURN_IF_ERROR(
                    NameRangesHelper(attrs, op_def.input_arg(), op_def, inputs));
        }
        if (outputs != nullptr) {
            return NameRangesHelper(attrs, op_def.output_arg(), op_def, outputs);
        }
        return Status::OK();
    }

    Status NameRangesForNode(const Node& node, const OpDef& op_def,
            NameRangeMap* inputs, NameRangeMap* outputs) {
        return NameRangesForNode(node.def(), op_def, inputs, outputs);
    }


    // definitions for all types
    Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            int32* value){

        const AttrValue* attr_value;
        TF_RETURN_IF_ERROR(attrs.Find(attr_name, &attr_value));
        TF_RETURN_IF_ERROR(AttrValueHasType(*attr_value, "int"));
        const auto& v = attr_value->i();
        // *value = v;
        // cast to
        *value = static_cast<int32>(v);
        return Status::OK();
    }

    Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            DataType* value){
        const AttrValue* attr_value;
        TF_RETURN_IF_ERROR(attrs.Find(attr_name, &attr_value));
        TF_RETURN_IF_ERROR(AttrValueHasType(*attr_value, "type"));
        const auto& v = attr_value->type();
        *value = v;
        return Status::OK();
    }

    Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            const TensorProto** value) {
        const AttrValue* attr_value;
        TF_RETURN_IF_ERROR(attrs.Find(attr_name, &attr_value));
        TF_RETURN_IF_ERROR(AttrValueHasType(*attr_value, "tensor"));
        *value = &attr_value->tensor();
        return Status::OK();
    }

    Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            bool* value){
        const AttrValue* attr_value;
        TF_RETURN_IF_ERROR(attrs.Find(attr_name, &attr_value));
        TF_RETURN_IF_ERROR(AttrValueHasType(*attr_value, "bool"));
        const auto& v = attr_value->b();
        *value = v;
        return Status::OK();
    }

    string FormatNodeForError(const Node& node){
        return node.name();
    }
    string FormatNodeDefForError(const NodeDef& node_def){
        return node_def.name();
    }

    Status AttachDef(const Status& status, const NodeDef& node_def,
            bool allow_multiple_formatted_node) {
        Status ret = status;
        string node_error;
        if (!allow_multiple_formatted_node &&
                status.error_message().find("{{node ") != string::npos) {
            node_error = node_def.name();
        } else {
            node_error = FormatNodeDefForError(node_def);
        }
        errors::AppendToMessage(&ret, strings::StrCat(" [[", node_error, "]]"));
        return ret;
    }

    Status AttachDef(const Status& status, const Node& node,
            bool allow_multiple_formatted_node) {
        return AttachDef(status, node.def(), allow_multiple_formatted_node);
    }

    bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            const TensorProto** value) {
        const AttrValue* attr_value = attrs.Find(attr_name);
        if (attr_value == nullptr) {
            return false;
        }
        Status s = AttrValueHasType(*attr_value, "tensor");
        if (!s.ok()) {
            return false;
        }
        *value = &attr_value->tensor();
        return true;
    }

    bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            std::vector<const TensorShapeProto*>* value) {
        const AttrValue* attr_value = attrs.Find(attr_name);
        if (attr_value == nullptr) {
            return false;
        }
        Status s = AttrValueHasType(*attr_value, "list(shape)");
        if (!s.ok()) {
            return false;
        }
        value->reserve(attr_value->list().shape().size());
        for (const auto& v : attr_value->list().shape()) {
            value->push_back(&v);
        }
        return true;
    }

    bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            std::vector<const string*>* value) {
        const AttrValue* attr_value = attrs.Find(attr_name);
        if (attr_value == nullptr) {
            return false;
        }
        Status s = AttrValueHasType(*attr_value, "list(string)");
        if (!s.ok()) {
            return false;
        }
        value->reserve(attr_value->list().s().size());
        for (const auto& v : attr_value->list().s()) {
            value->push_back(&v);
        }
        return true;
    }

    static const string& kEmptyString = *new string();

    const string& GetNodeAttrString(const AttrSlice& attrs, StringPiece attr_name) {
        const AttrValue* attr_value = attrs.Find(attr_name);
        if (attr_value == nullptr) {
            return kEmptyString;
        }
        Status s = AttrValueHasType(*attr_value, "string");
        if (!s.ok()) {
            return kEmptyString;
        }
        return attr_value->s();
    }

    #define DEFINE_TRY_GET_ATTR(TYPE, FIELD, ATTR_TYPE, APPEND_OP, CAST, ...) \
    bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,      \
                        TYPE* value) {                                      \
      const AttrValue* attr_value = attrs.Find(attr_name);                  \
      if (attr_value == nullptr) {                                          \
        return false;                                                       \
      }                                                                     \
      Status s = AttrValueHasType(*attr_value, ATTR_TYPE);                  \
      if (!s.ok()) {                                                        \
        return false;                                                       \
      }                                                                     \
      const auto& v = attr_value->FIELD();                                  \
      __VA_ARGS__;                                                          \
      *value = CAST;                                                        \
      return true;                                                          \
    }                                                                       \
    bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,      \
                        std::vector<TYPE>* value) {                         \
      const AttrValue* attr_value = attrs.Find(attr_name);                  \
      if (attr_value == nullptr) {                                          \
        return false;                                                       \
      }                                                                     \
      Status s = AttrValueHasType(*attr_value, "list(" ATTR_TYPE ")");      \
        if (!s.ok()) {                                                        \
        return false;                                                       \
      }                                                                     \
      value->reserve(attr_value->list().FIELD().size());                    \
      for (const auto& v : attr_value->list().FIELD()) {                    \
        __VA_ARGS__;                                                        \
        value->APPEND_OP(CAST);                                             \
      }                                                                     \
      return true;                                                          \
    }
    DEFINE_TRY_GET_ATTR(int64, i, "int", emplace_back, v, ;)
    DEFINE_TRY_GET_ATTR(
      int32, i, "int", emplace_back, static_cast<int32>(v),
      if (static_cast<int64>(static_cast<int32>(v)) != v) {
        static int log_counter = 0;
        if (log_counter < 10) {
          log_counter++;
          LOG(WARNING) << "Attr " << attr_name << " has value " << v
                       << " out of range for an int32";
        }
        return false;
      })
    DEFINE_TRY_GET_ATTR(float, f, "float", emplace_back, v, ;)
    DEFINE_TRY_GET_ATTR(bool, b, "bool", push_back, v, ;)

}// namespace dlxnet
