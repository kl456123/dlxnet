#ifndef DLXNET_CORE_FRAMEWORK_FUNCTION_H_
#define DLXNET_CORE_FRAMEWORK_FUNCTION_H_
#include "dlxnet/core/lib/core/status.h"
#include "dlxnet/core/framework/tensor.h"
#include "dlxnet/core/framework/types.h"
#include "dlxnet/core/framework/op.h"
#include "dlxnet/core/lib/gtl/inlined_vector.h"
#include "dlxnet/core/lib/gtl/array_slice.h"

namespace dlxnet{
    class CallFrameInterface {
        public:
            virtual ~CallFrameInterface() {}

            virtual size_t num_args() const = 0;
            virtual size_t num_retvals() const = 0;

            virtual Status GetArg(int index, Tensor* val) const = 0;
            virtual Status SetRetval(int index, const Tensor& val) = 0;
    };

    // Represents a function call frame. I.e., the data structure used to
    // pass arguments to a function and retrieve its results.
    //
    // Runtime must arrange accesses to one FunctionCallFrame s.t.
    //   1. SetArgs() happens before any GetArg();
    //   2. GetRetvals happens after all SetRetval();
    class FunctionCallFrame : public CallFrameInterface {
        public:
            FunctionCallFrame(DataTypeSlice arg_types, DataTypeSlice ret_types);
            ~FunctionCallFrame() override;

            // used outside
            // Caller methods.
            Status SetArgs(gtl::ArraySlice<Tensor> args);
            Status GetRetvals(std::vector<Tensor>* rets) const;

            // Moves the return values from the frame to rets. If allow_dead_tensors is
            // false it will fail if any of the retvals do not have a value.
            Status ConsumeRetvals(std::vector<Tensor>* rets, bool allow_dead_tensors);

            size_t num_args() const override { return arg_types_.size(); }
            size_t num_retvals() const override { return ret_types_.size(); }

            // internal used
            // Callee methods.
            Status GetArg(int index, Tensor* val) const override;
            Status SetRetval(int index, const Tensor& val) override;

        private:
            DataTypeVector arg_types_;
            DataTypeVector ret_types_;
            gtl::InlinedVector<Tensor, 4> args_;
            struct Retval {
                bool has_val = false;
                Tensor val;
            };
            gtl::InlinedVector<Retval, 4> rets_;

            TF_DISALLOW_COPY_AND_ASSIGN(FunctionCallFrame);
    };

    // Just A Functions Container
    // Helper to maintain a map between function names in a given
    // FunctionDefLibrary and function definitions.
    //
    // This class is thread-safe.
    class FunctionLibraryDefinition : public OpRegistryInterface {
        public:
            // Ops created for function arguments bear the name given by `kArgOp`; those
            // created for return values bear the name given by `kRetOp`.
            static constexpr const char* const kArgOp = "_Arg";
            static constexpr const char* const kDeviceArgOp = "_DeviceArg";
            static constexpr const char* const kRetOp = "_Retval";
            static constexpr const char* const kDeviceRetOp = "_DeviceRetval";
            static constexpr const char* const kIntsOnDeviceAttr =
                "experimental_ints_on_device";

            static constexpr const char* const kGradientOp = "SymbolicGradient";
            static constexpr const char* const kFuncAttr = "f";

            // OpRegistryInterface method. Useful for constructing a Graph.
            //
            // If "op" is defined in the library, returns its signature.
            // Otherwise, assume "op" is a primitive op and returns its op
            // signature and shape inference function.
            //
            // NB: This function outputs a borrowed pointer, which can be invalidated by a
            // subsequent call to `ReplaceFunction()` with the given name.
            Status LookUp(const string& op_type_name,
                    const OpRegistrationData** op_reg_data) const override;

            const OpRegistryInterface* default_registry() const {
                return default_registry_;
            }
        private:
            const OpRegistryInterface* const default_registry_;
    };
}


#endif
