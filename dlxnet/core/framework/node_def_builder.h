#ifndef DLXNET_CORE_FRAMEWORK_NODE_DEF_BUILDER_H_
#define DLXNET_CORE_FRAMEWORK_NODE_DEF_BUILDER_H_
#include "dlxnet/core/framework/types.h"
#include "dlxnet/core/framework/op_def.pb.h"
#include "dlxnet/core/framework/node_def.pb.h"
#include "dlxnet/core/framework/op.h"
#include "dlxnet/core/framework/node_def_util.h"
#include "dlxnet/core/framework/attr_value_util.h"
#include "dlxnet/core/lib/status.h"

namespace dlxnet{
    class NodeDefBuilder;


    class NodeDefBuilder{
        public:
            NodeDefBuilder(StringPiece name, StringPiece op_name,
                    const OpRegistryInterface* op_reistry=OpRegistry::Global());
            NodeDefBuilder(StringPiece name, const OpDef* op_def);

            NodeDefBuilder& Input(StringPiece src_node, int src_index,  DataType dt);
            NodeDefBuilder& Device(StringPiece device_spec);

            // set some attrs
            NodeDefBuilder& Attr(StringPiece name, float value);
            NodeDefBuilder& Attr(StringPiece name, int32 value);
            NodeDefBuilder& Attr(StringPiece name, const AttrValue& value);
            NodeDefBuilder& Attr(StringPiece name, AttrValue&& value);
            NodeDefBuilder& Attr(StringPiece name, const TensorProto& value);
            NodeDefBuilder& Attr(StringPiece name, const Tensor& value);

            Status Finalize(NodeDef* node_def);

            const string& node_name()const {return node_def_.name();}
            const OpDef& op_def()const{return *op_def_;}

        private:
            // init to construct
            void Initialize();
            const OpDef* op_def_;
            NodeDef node_def_;
            std::vector<string> errors_;
            int inputs_specified_;

    };
}



#endif
