#ifndef DLXNET_CORE_FRAMEWORK_FUNCTION_H_
#define DLXNET_CORE_FRAMEWORK_FUNCTION_H_
#include "dlxnet/core/lib/status.h"
#include "dlxnet/core/framework/tensor.h"
#include "dlxnet/core/lib/gtl/inlined_vector.h"
#include "dlxnet/core/lib/gtl/array_slice.h"
#include "dlxnet/core/framework/types.h"

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
}


#endif
