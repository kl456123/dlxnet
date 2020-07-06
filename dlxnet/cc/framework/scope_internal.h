#ifndef DLXNET_CC_FRAMEWORK_SCOPE_INTERNAL_H_
#define DLXNET_CC_FRAMEWORK_SCOPE_INTERNAL_H_
#include "dlxnet/cc/framework/scope.h"


namespace dlxnet{

    class ShapeRefiner;
    class Scope::Impl {
        public:
            // A NameMap is used to keep track of suffixes for names used in a scope. A
            // name that has not been used so far in a scope will get no suffix. Later
            // uses of the same name will get suffixes _1, _2, _3, etc. Multiple scopes
            // can share the same NameMap. For instance, a new scope created using
            // WithControlDependencies() would share the same NameMap with the parent.
            typedef std::unordered_map<string, int> NameMap;

            Impl(const std::shared_ptr<Graph>& graph,
                    const std::shared_ptr<Status>& status,
                    const std::shared_ptr<NameMap>& name_map,
                    const std::shared_ptr<ShapeRefiner>& refiner);

            const string& name() const { return name_; }

        private:
            friend class Scope;

            // Tag types to choose the constructor to dispatch.
            struct Tags {
                enum class ScopeName;
                enum class OpName;
                enum class Device;
                enum class AssignedDevice;
            };

            Impl(Graph* graph, Status* status, NameMap* name_map, ShapeRefiner* refiner);
            Impl(const Scope& other, Tags::ScopeName, const string& name,
                    bool copy_names);
            Impl(const Scope& other, Tags::OpName, const string& name,
                    const string& op_name);
            Impl(const Scope& other, Tags::Device, const string& device);
            Impl(const Scope& other, Tags::AssignedDevice, const string& assigned_device);

            // Helper functions to get a unique names.
            string GetUniqueName(const string& prefix) const;


            // The graph, status, and name maps are shared by all child scopes
            // created from a single 'root' scope. A root scope is created by calling the
            // Scope::NewRootScope function, which creates a new graph, a new status and
            // the name maps.
            std::shared_ptr<Graph> graph_ = nullptr;
            std::shared_ptr<Status> status_ = nullptr;
            std::shared_ptr<NameMap> name_map_ = nullptr;
            std::shared_ptr<ShapeRefiner> refiner_ = nullptr;



            // The fully-qualified name of this scope (i.e. includes any parent scope
            // names).
            const string name_ = "";
            const string op_name_ = "";
            const string device_ = "";
            const string assigned_device_ = "";
    };

}// namespace dlxnet


#endif
