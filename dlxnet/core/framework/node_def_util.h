#ifndef DLXNET_CORE_FRAMEWORK_NODE_DEF_UTIL_H_
#define DLXNET_CORE_FRAMEWORK_NODE_DEF_UTIL_H_
#include "dlxnet/core/platform/protobuf.h"
#include "dlxnet/core/framework/op_def.pb.h"
#include "dlxnet/core/framework/node_def.pb.h"
#include "dlxnet/core/framework/types.h"
#include "dlxnet/core/lib/stringpiece.h"
#include "dlxnet/core/lib/status.h"
#include "dlxnet/core/lib/gtl/flatmap.h"



namespace dlxnet{
    class Node;
    class NodeDef;
    class OpDef;
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
    Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            int32* value);  // type: "int"
    Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            DataType* value);// type: "DataType"
    Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
            const TensorProto** value);


}


#endif
