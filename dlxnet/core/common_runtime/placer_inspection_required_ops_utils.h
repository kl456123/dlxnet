#ifndef DLXNET_CORE_COMMON_RUNTIME_PLACER_INSPECTION_REQUIRED_OPS_UTILS_H_
#define DLXNET_CORE_COMMON_RUNTIME_PLACER_INSPECTION_REQUIRED_OPS_UTILS_H_
#include "dlxnet/core/platform/types.h"
#include "dlxnet/core/graph/graph.h"

namespace dlxnet{

    // The "call" stack of functions.
    // Useful for better error messages as well as for detecting recursion.
    // Stores references to graph nodes. These references must outlive this.
    class FunctionStack {
        public:
            explicit FunctionStack(const string& function_name);

            // `node_in_current_function` must outlive this.
            FunctionStack Push(const Node* node_in_current_function,
                    const string& new_current_function) const;

            const string& current_function_name() const { return current_function_name_; }

        private:
            struct Frame {
                Frame(const string& function, const Node* node)
                    : function_name(function), node(node) {}

                string function_name;
                const Node* node;
            };

            // The function at the top of the stack. In other words, the function
            // that is currently being inspected for placement.
            string current_function_name_;

            // The stack of frames that got the placement to the current_function_name_.
            // frames_[0].function_name is the top function that Placer was constructed
            // with. frames_[0].function_name can be empty if placer was constructed with
            // a nameless graph, not a function.  frames_[0].node_name is a name of a node
            // in frames_[0].function_name that required deep inspection (e.g. a
            // PartitionedCallOp). The function that this node invoked is
            // frames_[1].function_name, if frames_.size() > 1.  Else, the function that
            // this node invoked is current_function_name_.
            std::vector<Frame> frames_;
    };
} // namespace dlxnet


#endif
