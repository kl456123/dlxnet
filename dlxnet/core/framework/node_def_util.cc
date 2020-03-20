#include "dlxnet/core/framework/node_def_util.h"


namespace dlxnet{
    AttrSlice::AttrSlice(const NodeDef& node_def)
        :ndef_(&node_def), attrs_(&ndef_->attr()){}

    string AttrSlice::DebugString()const{
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
}
