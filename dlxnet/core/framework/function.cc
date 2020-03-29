#include "dlxnet/core/framework/function.h"


namespace dlxnet{
    FunctionCallFrame::FunctionCallFrame(DataTypeSlice arg_types,
            DataTypeSlice ret_types)
        : arg_types_(arg_types.begin(), arg_types.end()),
        ret_types_(ret_types.begin(), ret_types.end()) {
            args_.resize(arg_types_.size());
            rets_.resize(ret_types_.size());
        }
    FunctionCallFrame::~FunctionCallFrame() {}

    Status FunctionCallFrame::SetArgs(gtl::ArraySlice<Tensor> args) {
        // check input types and nums
        // Input type checks.
        if (args.size() != arg_types_.size()) {
            return errors::InvalidArgument("Expects ", arg_types_.size(),
                    " arguments, but ", args.size(),
                    " is provided");
        }
        for (size_t i = 0; i < args.size(); ++i) {
            if (arg_types_[i] != args[i].dtype()) {
                return errors::InvalidArgument(
                        "Expects arg[", i, "] to be ", DataTypeString(arg_types_[i]), " but ",
                        DataTypeString(args[i].dtype()), " is provided");
            }
            args_[i] = args[i];
        }
        return Status::OK();
    }
    Status FunctionCallFrame::GetRetvals(std::vector<Tensor>* rets) const {
        rets->clear();
        rets->reserve(rets_.size());
        for (size_t i = 0; i < rets_.size(); ++i) {
            const auto& item = rets_[i];
            if (item.has_val) {
                rets->push_back(item.val);
            } else {
                return errors::Internal("Retval[", i, "] does not have value");
            }
        }

    }
    Status FunctionCallFrame::ConsumeRetvals(std::vector<Tensor>* rets,
            bool allow_dead_tensors) {
        rets->clear();
        rets->reserve(rets_.size());
        for (size_t i = 0; i < rets_.size(); ++i) {
            if (rets_[i].has_val) {
                rets->emplace_back(std::move(rets_[i].val));
            } else if (allow_dead_tensors) {
                rets->emplace_back();
            } else {
                return errors::Internal("Retval[", i, "] does not have value");
            }
        }
        return Status::OK();
    }

    Status FunctionCallFrame::GetArg(int index, Tensor* val) const {
        if (index < 0 || static_cast<size_t>(index) >= args_.size()) {
            return errors::InvalidArgument("GetArg ", index, " is not within [0, ",
                    args_.size(), ")");
        }
        *val = args_[index];
        return Status::OK();
    }
    Status FunctionCallFrame::SetRetval(int index, const Tensor& val) {
        if (index < 0 || static_cast<size_t>(index) >= rets_.size()) {
            return errors::InvalidArgument("SetRetval ", index, " is not within [0, ",
                    rets_.size(), ")");
        }
        if (val.dtype() != ret_types_[index]) {
            return errors::InvalidArgument(
                    "Expects ret[", index, "] to be ", DataTypeString(ret_types_[index]),
                    ", but ", DataTypeString(val.dtype()), " is provided.");
        }
        Retval* item = &rets_[index];
        if (!item->has_val) {
            item->has_val = true;
            item->val = val;
        } else {
            return errors::Internal("Retval[", index, "] has already been set.");
        }
        return Status::OK();
    }
}// namespace dlxnet
