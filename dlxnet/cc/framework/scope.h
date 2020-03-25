#ifndef DLXNET_CC_FRAMEWORK_SCOPE_H_
#define DLXNET_CC_FRAMEWORK_SCOPE_H_
#include <memory>

#include "dlxnet/core/common_runtime/shape_refiner.h"
#include "dlxnet/core/framework/graph.pb.h"
#include "dlxnet/core/graph/graph.h"


namespace dlxnet{
    // forward declaration
    class NodeBuilder;
    class Graph;
    class GraphDef;

    class Scope{
        public:
            Scope(const Scope& other);
            ~Scope();
            Scope& operator=(const Scope& other);

            // The following functions are for users making graphs. They return brand new
            // scopes, or scopes derived from an existing scope object.

            /// Return a new scope.
            /// This creates a new graph and all operations constructed in this graph
            /// should use the returned object as the "root" scope.
            static Scope NewRootScope();

            /// Return a new scope. Ops created with this scope will have
            /// `name/child_scope_name` as the prefix. The actual name will be unique
            /// in the current scope. All other properties are inherited from the current
            /// scope. If `child_scope_name` is empty, the `/` is elided.
            Scope NewSubScope(const string& child_scope_name) const;

            /// Return a new scope. All ops created within the returned scope will have
            /// names of the form `name/StrCat(fragments...)[_suffix]`
            template <typename... Ty>
                Scope WithOpName(Ty... fragments) const {
                    return WithOpNameImpl(absl::StrCat(fragments...));
                }

            /// If status() is Status::OK(), convert the Graph object stored in this scope
            /// to a GraphDef proto and return Status::OK(). Otherwise, return the error
            /// status as is without performing GraphDef conversion.
            Status ToGraphDef(GraphDef* gdef) const;


            // Calls AddNode() using this scope's ShapeRefiner. This exists in the public
            // API to prevent custom op wrappers from needing access to shape_refiner.h or
            // scope_internal.h.
            // TODO(skyewm): remove this from public API
            Status DoShapeInference(Node* node) const;

            /// Update the status on this scope.
            /// Note: The status object is shared between all children of this scope.
            /// If the resulting status is not Status::OK() and exit_on_error_ is set on
            /// this scope, this function exits by calling LOG(FATAL).
            void UpdateStatus(const Status& s) const;

            // START_SKIP_DOXYGEN

            /// Update the builder with properties accumulated in this scope. Does not set
            /// status().
            // TODO(skyewm): NodeBuilder is not part of public API
            void UpdateBuilder(NodeBuilder* builder) const;
            // END_SKIP_DOXYGEN
            //
            bool ok() const;

            string GetUniqueNameForOp(const string& default_name)const;

            // TODO(skyewm): Graph is not part of public API
            Graph* graph() const;

            Status status() const;

            // accessor
            // START_SKIP_DOXYGEN
            class Impl;
            Impl* impl() { return impl_.get(); }
            const Impl* impl() const { return impl_.get(); }
            // END_SKIP_DOXYGEN

        private:
            Scope WithOpNameImpl(const string& op_name) const;
            std::unique_ptr<Impl> impl_;
            explicit Scope(Impl*);

    };
}

#endif

