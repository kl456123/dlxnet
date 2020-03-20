#ifndef DLXNET_CORE_FRAMEWORK_NODE_DEF_UTIL_H_
#define DLXNET_CORE_FRAMEWORK_NODE_DEF_UTIL_H_
#include "dlxnet/core/platform/protobuf.h"
#include "dlxnet/core/framework/op_def.pb.h"
#include "dlxnet/core/framework/node_def.pb.h"
#include "dlxnet/core/lib/stringpiece.h"
#include "dlxnet/core/lib/status.h"



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

    typedef protobuf::Map<string, AttrValue> AttrValueMap;


    class AttrSlice{
        public:
            AttrSlice(const NodeDef& node_def);

            const AttrValue* Find(StringPiece attr_name)const;

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

    // Validates that the NodeDef:
    // * Defines all expected attrs from the OpDef.
    // * All attrs satisfies constraints from the OpDef.
    // * Has a signature matching SignatureForNode().
    // etc.
    Status ValidateNodeDef(const NodeDef& node_def, const OpDef& op_def);


}


#endif
