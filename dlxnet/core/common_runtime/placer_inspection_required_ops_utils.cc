#include "dlxnet/core/common_runtime/placer_inspection_required_ops_utils.h"

namespace dlxnet{
    FunctionStack::FunctionStack(const string& function_name)
        : current_function_name_(function_name) {}

    FunctionStack FunctionStack::Push(const Node* node_in_current_function,
            const string& new_current_function) const {
        FunctionStack new_stack(new_current_function);
        new_stack.frames_ = frames_;
        new_stack.frames_.emplace_back(current_function_name_,
                node_in_current_function);
        return new_stack;
    }
} // namespace dlxnet
