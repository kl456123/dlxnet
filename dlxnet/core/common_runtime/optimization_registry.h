#ifndef DLXNET_CORE_COMMON_RUNTIME_OPTIMIZATION_REGISTRY_H_
#define DLXNET_CORE_COMMON_RUNTIME_OPTIMIZATION_REGISTRY_H_
#include <unordered_map>
#include <map>
#include <memory>

#include "dlxnet/core/common_runtime/device_set.h"
#include "dlxnet/core/graph/graph.h"


namespace dlxnet{
    struct SessionOptions;
    // All the parameters used by an optimization pass are packaged in
    // this struct. They should be enough for the optimization pass to use
    // as a key into a state dictionary if it wants to keep state across
    // calls.
    struct GraphOptimizationPassOptions {
        // Filled in by DirectSession for PRE_PLACEMENT optimizations. Can be empty.
        string session_handle;
        const SessionOptions* session_options = nullptr;

        // The DeviceSet contains all the devices known to the system and is
        // filled in for optimizations run by the session master, i.e.,
        // PRE_PLACEMENT, POST_PLACEMENT, and POST_REWRITE_FOR_EXEC. It is
        // nullptr for POST_PARTITIONING optimizations which are run at the
        // workers.
        const DeviceSet* device_set = nullptr;  // Not owned.

        // The graph to optimize, for optimization passes that run before
        // partitioning. Null for post-partitioning passes.
        // An optimization pass may replace *graph with a new graph object.
        std::unique_ptr<Graph>* graph = nullptr;

        // Graphs for each partition, if running post-partitioning. Optimization
        // passes may alter the graphs, but must not add or remove partitions.
        // Null for pre-partitioning passes.
        std::unordered_map<string, std::unique_ptr<Graph>>* partition_graphs =
            nullptr;
    };

    // Optimization passes are implemented by inheriting from
    // GraphOptimizationPass.
    class GraphOptimizationPass {
        public:
            virtual ~GraphOptimizationPass() {}
            virtual Status Run(const GraphOptimizationPassOptions& options) = 0;
            void set_name(const string& name) { name_ = name; }
            string name() const { return name_; }

        private:
            // The name of the opitimization pass, which is the same as the inherited
            // class name.
            string name_;
    };

    // The key is a 'phase' number. Phases are executed in increasing
    // order. Within each phase the order of passes is undefined.
    typedef std::map<int, std::vector<std::unique_ptr<GraphOptimizationPass>>>
        GraphOptimizationPasses;

    class OptimizationPassRegistry{
        public:
            // Groups of passes are run at different points in initialization.
            enum Grouping {
                PRE_PLACEMENT,          // after cost model assignment, before placement.
                POST_PLACEMENT,         // after placement.
                POST_REWRITE_FOR_EXEC,  // after re-write using feed/fetch endpoints.
                POST_PARTITIONING,      // after partitioning
            };

            // Add an optimization pass to the registry.
            void Register(Grouping grouping, int phase,
                    std::unique_ptr<GraphOptimizationPass> pass);

            const std::map<Grouping, GraphOptimizationPasses>& groups() {
                return groups_;
            }

            // Run all passes in grouping, ordered by phase, with the same
            // options.
            Status RunGrouping(Grouping grouping,
                    const GraphOptimizationPassOptions& options);

            // Returns the global registry of optimization passes.
            static OptimizationPassRegistry* Global();

            // Prints registered optimization passes for debugging.
            void LogGrouping(Grouping grouping, int vlog_level);
            void LogAllGroupings(int vlog_level);

        private:
            std::map<Grouping, GraphOptimizationPasses> groups_;
    };

    namespace optimization_registration {

        class OptimizationPassRegistration {
            public:
                OptimizationPassRegistration(OptimizationPassRegistry::Grouping grouping,
                        int phase,
                        std::unique_ptr<GraphOptimizationPass> pass,
                        string optimization_pass_name) {
                    pass->set_name(optimization_pass_name);
                    OptimizationPassRegistry::Global()->Register(grouping, phase,
                            std::move(pass));
                }
        };

    }  // namespace optimization_registration

#define REGISTER_OPTIMIZATION(grouping, phase, optimization) \
    REGISTER_OPTIMIZATION_UNIQ_HELPER(__COUNTER__, grouping, phase, optimization)

#define REGISTER_OPTIMIZATION_UNIQ_HELPER(ctr, grouping, phase, optimization) \
    REGISTER_OPTIMIZATION_UNIQ(ctr, grouping, phase, optimization)

#define REGISTER_OPTIMIZATION_UNIQ(ctr, grouping, phase, optimization)         \
    static ::dlxnet::optimization_registration::OptimizationPassRegistration \
    register_optimization_##ctr(                                             \
            grouping, phase,                                                     \
            ::std::unique_ptr<::dlxnet::GraphOptimizationPass>(              \
                new optimization()),                                             \
#optimization)
}// namespace dlxnet


#endif
