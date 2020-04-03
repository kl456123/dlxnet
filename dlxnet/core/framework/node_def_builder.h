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

            // Sets the attr, if not already set.  If already set with a different
            // value, an error will be returned from Finalize().
            NodeDefBuilder& Attr(StringPiece name, const AttrValue& value);
            NodeDefBuilder& Attr(StringPiece name, AttrValue&& value);
            NodeDefBuilder& Attr(StringPiece name, StringPiece value);
            NodeDefBuilder& Attr(StringPiece name, const char* value);
            NodeDefBuilder& Attr(StringPiece name, int32 value);
            NodeDefBuilder& Attr(StringPiece name, int64 value);
            NodeDefBuilder& Attr(StringPiece name, float value);
            NodeDefBuilder& Attr(StringPiece name, double value);
            NodeDefBuilder& Attr(StringPiece name, bool value);
            NodeDefBuilder& Attr(StringPiece name, DataType value);
            NodeDefBuilder& Attr(StringPiece name, const Tensor& value);
            NodeDefBuilder& Attr(StringPiece name, const TensorProto& value);

            Status Finalize(NodeDef* node_def);

            const string& node_name()const {return node_def_.name();}
            const OpDef& op_def()const{return *op_def_;}



        private:
            // for internal usage purpose

            // Get the current ArgDef and advance to the next one. Returns nullptr
            // if no more inputs are available.
            const OpDef::ArgDef* NextArgDef();

            // Returns true if there is still an input_arg available in *op_def_,
            // otherwise adds to error_ and returns false.
            bool NextArgAvailable();
            // Returns true if an attr named `name` is already present in the node_def_.
            // If such an attr is already present and `value` is not equal to the present
            // value, an error is generated.
            bool AttrValueAlreadyPresent(StringPiece name, const AttrValue& value);

            void AddInput(StringPiece src_node, int src_index);

            // These do the main work of the Input() methods.
            void SingleInput(const OpDef::ArgDef* input_arg, StringPiece src_node,
                    int src_index, DataType dt);
            // Makes dt a ref type if that is what the input_arg specifies.
            DataType MaybeAddRef(const OpDef::ArgDef* input_arg, DataType dt) {
                return input_arg->is_ref() ? MakeRefType(dt) : dt;
            }

            // Generate an error if you can't pass dt when expected is expected.
            void VerifyInputType(const OpDef::ArgDef* input_arg, DataType expected,
                    DataType dt);

            // If input_arg->is_ref() is true, generate an error if dt is not a ref.
            void VerifyInputRef(const OpDef::ArgDef* input_arg, DataType dt);

            // init to construct
            void Initialize();
            const OpDef* op_def_;
            NodeDef node_def_;
            std::vector<string> errors_;
            int inputs_specified_;

    };
}



#endif
