#ifndef DLXNET_CORE_FRAMEWORK_NODE_DEF_UTIL_H_
#define DLXNET_CORE_FRAMEWORK_NODE_DEF_UTIL_H_
#include "dlxnet/core/platform/protobuf.h"
#include "dlxnet/core/framework/op_def.pb.h"
#include "dlxnet/core/framework/node_def.pb.h"


namespace dlxnet{
    // manage node attrs
    class AttrSlice;

    // summarize node and node_def

    string SummarizeNode();
    string SummarizeNodeDef();

    string SummarizeAttrs();

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

    void AddDefaultToNodeDef(const OpDef& op_def, NodeDef& node_def);

    void AddNodeAttr(StringPiece name, const AttrValue& value, NodeDef* node_def);


}


#endif
