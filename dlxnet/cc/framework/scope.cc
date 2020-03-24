#include "dlxnet/cc/framework/scope.h"
#include "dlxnet/cc/framework/scope_internal.h"
#include "dlxnet/core/graph/node_builder.h"


namespace dlxnet{
    // some copy and assign functions
    Scope::Scope(Impl* impl) : impl_(impl) {}

    Scope::Scope(const Scope& other) : impl_(new Impl(*other.impl())) {}

    Scope::~Scope() {}

    Scope& Scope::operator=(const Scope& other) {
        // We can't copy Impls because of the const members, use copy ctor instead
        impl_.reset(new Impl(*other.impl_));
        return *this;
    }

    namespace {
        // sperator used to generate unique name
        const char kSuffixSeparator[] = "_";
        const char kScopeSeparator[] = "/";
    }// namespace

    Scope::Impl::Impl(Graph* graph, Status* status, NameMap* name_map,
            ShapeRefiner* refiner)
        : graph_(graph),
        status_(status),
        name_map_(name_map),
        refiner_(refiner){}

    Scope::Impl::Impl(const std::shared_ptr<Graph>& graph,
            const std::shared_ptr<Status>& status,
            const std::shared_ptr<NameMap>& name_map,
            const std::shared_ptr<ShapeRefiner>& refiner)
        : graph_(graph),
        status_(status),
        name_map_(name_map),
        refiner_(refiner){}

    /*static */ Scope Scope::NewRootScope() {
        Graph* graph = new Graph(OpRegistry::Global());
        ShapeRefiner* refiner =
            new ShapeRefiner(graph->op_registry());
        return Scope(new Impl(graph, new Status, new Impl::NameMap, refiner));
    }

    // some types of pseudo copy constructors, dispatch when used
    Scope::Impl::Impl(const Scope& other, Tags::ScopeName, const string& name,
            bool copy_names)
        : graph_(other.impl()->graph_),
        status_(other.impl()->status_),
        name_map_(copy_names ? other.impl()->name_map_
                : std::shared_ptr<NameMap>(new NameMap)),
        refiner_(other.impl()->refiner_),
        name_(name),
        op_name_(""),
        device_(other.impl()->device_){}


    Scope::Impl::Impl(const Scope& other, Tags::OpName, const string& name,
            const string& op_name)
        : graph_(other.impl()->graph_),
        status_(other.impl()->status_),
        name_map_(other.impl()->name_map_),
        refiner_(other.impl()->refiner_),
        name_(name),
        op_name_(op_name),
        device_(other.impl()->device_){}


    bool Scope::ok() const { return impl()->status_->ok(); }

    Graph* Scope::graph() const { return impl()->graph_.get(); }
    Status Scope::status() const { return *impl()->status_; }

    // update status during construction
    void Scope::UpdateStatus(const Status& s) const {
        impl()->status_->Update(s);
        if (!ok()) {
            LOG(FATAL) << *impl()->status_;
        }
    }

    // dispatch to graph
    Status Scope::ToGraphDef(GraphDef* gdef) const {
        if (!ok()) {
            return *impl()->status_;
        }
        graph()->ToGraphDef(gdef);
        return Status::OK();
    }

    void Scope::UpdateBuilder(NodeBuilder* builder) const {
        // add some attributions from local scope to current builder(current node the same)
        if (!impl()->device_.empty()) {
            builder->Device(impl()->device_);
        }
    }

    string Scope::Impl::GetUniqueName(const string& prefix) const {
        // duplication check
        auto entry = name_map_->find(prefix);
        if (entry == name_map_->end()) {
            name_map_->insert({prefix, 0});
            return prefix;
        }
        string unique_name;
        do {
            unique_name = strings::StrCat(prefix, kSuffixSeparator, ++entry->second);
        } while (name_map_->find(unique_name) != name_map_->end());
        name_map_->insert({unique_name, 0});
        return unique_name;
    }

    /*static*/ Scope Scope::NewSubScope(const string& child_scope_name) const {
        // if child name is empty, copy from parent
        if (child_scope_name.empty()) {
            return Scope(new Impl(*this, Impl::Tags::ScopeName(), impl()->name_,
                        true /* copy_names */));
        }
        const string unique_name =
            impl()->GetUniqueName(child_scope_name);

        // if parent name is empty
        const string sep =
            impl()->name_.empty() || unique_name.empty() ? "" : kScopeSeparator;
        return Scope(new Impl(*this, Impl::Tags::ScopeName(),
                    strings::StrCat(impl()->name_, sep, unique_name),
                    false /* copy_names */));
    }

    // single input version
    Scope Scope::WithOpNameImpl(const string& op_name) const {
        return Scope(new Impl(*this, Impl::Tags::OpName(), impl()->name_, op_name));
    }

    Status Scope::DoShapeInference(Node* node) const {
        return impl_->refiner_->AddNode(node);
    }




}// namespace dlxnet
