#ifndef DLXNET_CORE_FRAMEWORK_NODE_DEF_UTIL_H_
#define DLXNET_CORE_FRAMEWORK_NODE_DEF_UTIL_H_
#include "dlxnet/core/platform/protobuf.h"
#include "dlxnet/core/framework/op_def.pb.h"
#include "dlxnet/core/framework/node_def.pb.h"
#include "dlxnet/core/framework/types.h"
#include "dlxnet/core/lib/core/stringpiece.h"
#include "dlxnet/core/lib/core/status.h"
#include "dlxnet/core/lib/gtl/flatmap.h"
#include "dlxnet/core/framework/attr_value_util.h"



namespace dlxnet{
    class Node;
    class NodeDef;
    class OpDef;
    class DeviceMgr;
    // manage node attrs
    class AttrSlice;

    // summarize node and node_def

    string SummarizeNode(const Node& node);
    string SummarizeNodeDef(const NodeDef& node_def);

    string SummarizeAttrs(const NodeDef& node_def);
    string SummarizeAttrsHelper(AttrSlice attrs, StringPiece device);

    typedef protobuf::Map<string, AttrValue> AttrValueMap;


    class AttrSlice{
        public:
            AttrSlice(const NodeDef& node_def);

            AttrSlice();  // Empty
            explicit AttrSlice(const AttrValueMap* a);

            // two versions function of find
            const AttrValue* Find(StringPiece attr_name)const;
            // Returns the attr_value for attr_name if found. Otherwise, returns a
            // NotFound status.
            Status Find(StringPiece attr_name, const AttrValue** attr_value) const;

            int size()const {return attrs_->size();}
            string DebugString()const;
            // iteration over all attrs
            AttrValueMap::const_iterator begin()const{return attrs_->begin();}
            AttrValueMap::const_iterator end()const{return attrs_->end();}
        private:
            const NodeDef* ndef_;
            const AttrValueMap* attrs_;

    };

    // Return true if the attr with the name attr_name is defined in node_def.
    bool HasNodeAttr(const NodeDef& node_def, StringPiece attr_name);

    void AddDefaultsToNodeDef(const OpDef& op_def, NodeDef* node_def);

    void AddNodeAttr(StringPiece name, const AttrValue& value, NodeDef* node_def);
    void AddNodeAttr(StringPiece name, AttrValue&& value, NodeDef* node_def);

    // Validates that the NodeDef:
    // * Defines all expected attrs from the OpDef.
    // * All attrs satisfies constraints from the OpDef.
    // * Has a signature matching SignatureForNode().
    // etc.
    Status ValidateNodeDef(const NodeDef& node_def, const OpDef& op_def);

    // Computes the input and output types for a specific node.
    // REQUIRES: ValidateOpDef(op_def).ok()
    Status InOutTypesForNode(const NodeDef& node_def, const OpDef& op_def,
            DataTypeVector* inputs, DataTypeVector* outputs);

    // Computes the input types for a specific node.
    // REQUIRES: ValidateOpDef(op_def).ok()
    Status InputTypesForNode(const NodeDef& node_def, const OpDef& op_def,
            DataTypeVector* inputs);

    // Computes the output type for a specific node.
    // REQUIRES: ValidateOpDef(op_def).ok()
    Status OutputTypesForNode(const NodeDef& node_def, const OpDef& op_def,
            DataTypeVector* inputs);

    // both attrslice and node are ok
    typedef gtl::FlatMap<StringPiece, std::pair<int, int>, hash<StringPiece>>
        NameRangeMap;
    Status NameRangesForNode(const AttrSlice& attrs, const OpDef& op_def,
            NameRangeMap* inputs, NameRangeMap* outputs);
    Status NameRangesForNode(const Node& node, const OpDef& op_def,
            NameRangeMap* inputs, NameRangeMap* outputs);


    // functions to Get Node Attrs
    // declarations for all types
    // Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            // int32* value);  // type: "int"
    // Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            // DataType* value);// type: "DataType"
    // Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            // const TensorProto** value);
    // Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            // bool* value);

    // Look up the attr with name attr_name and set *value to its value.  If no
  // attr with attr_name is found in node_def, or the attr does not have
  // a matching type, a non-ok status will be returned.
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                     string* value);  // type: "string"
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                     tstring* value);  // type: "tstring"
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                     int64* value);  // type: "int"
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                     int32* value);  // type: "int"
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                     float* value);  // type: "float"
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                     bool* value);  // type: "bool"
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                     DataType* value);  // type: "type"
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                     TensorShapeProto* value);  // type: "shape"
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                     TensorShape* value);  // type: "shape"
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                     Tensor* value);  // type: "tensor"
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                     std::vector<string>* value);  // type "list(string)"
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                     std::vector<tstring>* value);  // type "list(tstring)"
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                     std::vector<int64>* value);  // type "list(int)"
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                     std::vector<int32>* value);  // type "list(int)"
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                     std::vector<float>* value);  // type "list(float)"
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                     std::vector<bool>* value);  // type "list(bool)"
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                     std::vector<DataType>* value);  // type "list(type)"
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                     DataTypeVector* value);  // type "list(type)"
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                     std::vector<TensorShapeProto>* value);  // type "list(shape)"
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                     std::vector<TensorShape>* value);  // type "list(shape)"

    // Produces a formatted string pattern from the node which can uniquely identify
    // this node upstream to produce an informative error message. The pattern
    // followed is: {{node <node_name>}}
    string FormatNodeForError(const Node& node);
    string FormatNodeDefForError(const NodeDef& node_def);

    // Returns "status" with formatted NodeDef attached as additional text
    // in the error message. If 'allow_multiple_formatted_node' is false and there
    // is already a formatted NodeDef present in 'status', we simply attach the name
    // of the NodeDef instead of the formatted string.
    Status AttachDef(const Status& status, const NodeDef& node_def,
            bool allow_multiple_formatted_node = false);
    Status AttachDef(const Status& status, const Node& node,
            bool allow_multiple_formatted_node = false);

    // Look up the attr with name attr_name and return a reference to its value.
    // If no attr with attr_name is found in node_def, or the attr does not have
    // a matching type, a reference to an empty string is returned.
    // REQUIRES: Must not use the returned value beyond the lifetime of node_def.
    const string& GetNodeAttrString(const AttrSlice& attrs, StringPiece attr_name);

    // This version avoids copying the TensorProto.
    // REQUIRES: Must not use *value beyond the lifetime of node_def.
    Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            const TensorProto** value);  // type: "tensor"
    bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            const TensorProto** value);  // type: "tensor"

    // This version avoids copying the NameAttrList.
    // REQUIRES: Must not use *value beyond the lifetime of node_def.
    Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            const NameAttrList** value);  // type: "func"
    bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            const NameAttrList** value);  // type: "func"

    // These versions copies the NameAttrList(s).
    Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            NameAttrList* value);  // type: "func"
    Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            std::vector<NameAttrList>* value);  // type: "list(func)"

    // Look up the attr with name attr_name and set *value to its value.  If no
    // attr with attr_name is found in node_def, or the attr does not have
    // a matching type, false is returned.
    bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            string* value);  // type: "string"
    bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            int64* value);  // type: "int"
    bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            std::vector<int64>* value);  // type: "int"
    bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            int32* value);  // type: "int"
    bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            float* value);  // type: "float"
    bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            bool* value);  // type: "bool"
    bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            DataType* value);  // type: "type"
    bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            TensorShape* value);  // type: "shape"

    bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            std::vector<string>* value);  // type: "list(string)"

    bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            std::vector<int32>* value);  // type: "list(int)"
    bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            std::vector<float>* value);  // type: "list(float)"
    bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            std::vector<bool>* value);  // type: "list(bool)"
    bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            std::vector<DataType>* value);  // type: "list(type)"
    bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            std::vector<TensorShape> value);  // type: "shape"
}


#endif
