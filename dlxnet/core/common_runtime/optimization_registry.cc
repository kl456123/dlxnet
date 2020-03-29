#include "dlxnet/core/common_runtime/optimization_registry.h"
#include "dlxnet/core/platform/env.h"


namespace dlxnet{
    // static
    OptimizationPassRegistry* OptimizationPassRegistry::Global() {
        static OptimizationPassRegistry* global_optimization_registry =
            new OptimizationPassRegistry;
        return global_optimization_registry;
    }

    void OptimizationPassRegistry::Register(
            Grouping grouping, int phase, std::unique_ptr<GraphOptimizationPass> pass) {
        groups_[grouping][phase].push_back(std::move(pass));
    }

    Status OptimizationPassRegistry::RunGrouping(
            Grouping grouping, const GraphOptimizationPassOptions& options) {
        auto group = groups_.find(grouping);
        if (group != groups_.end()) {
            for (auto& phase : group->second) {
                VLOG(1) << "Running optimization phase " << phase.first;
                for (auto& pass : phase.second) {
                    VLOG(1) << "Running optimization pass: " << pass->name();
                    const uint64 start_us = Env::Default()->NowMicros();
                    Status s = pass->Run(options);
                    const uint64 end_us = Env::Default()->NowMicros();
                    if (!s.ok()) return s;
                }
            }
        }
        return Status::OK();
    }

    void OptimizationPassRegistry::LogGrouping(Grouping grouping, int vlog_level) {
        auto group = groups_.find(grouping);
        if (group != groups_.end()) {
            for (auto& phase : group->second) {
                for (auto& pass : phase.second) {
                    VLOG(vlog_level) << "Registered optimization pass grouping " << grouping
                        << " phase " << phase.first << ": " << pass->name();
                }
            }
        }
    }

    void OptimizationPassRegistry::LogAllGroupings(int vlog_level) {
        for (auto group = groups_.begin(); group != groups_.end(); ++group) {
            LogGrouping(group->first, vlog_level);
        }
    }
}// namespace dlxnet
