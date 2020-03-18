#ifndef DLXNET_CORE_COMMON_RUNTIME_DIRECT_SESSION_H_
#define DLXNET_CORE_COMMON_RUNTIME_DIRECT_SESSION_H_
#include <vector>

#include "dlxnet/core/public/session.h"
#include "dlxnet/core/public/session_options.h"
#include "dlxnet/core/common_runtime/device_mgr.h"
#include "dlxnet/core/common_runtime/executor.h"
#include "dlxnet/core/common_runtime/graph_execution_state.h"
#include "dlxnet/core/lib/status.h"
#include "dlxnet/core/lib/stringpiece.h"
#include "dlxnet/core/lib/core/threadpool.h"
#include "dlxnet/core/platform/mutex.h"
#include "dlxnet/core/graph/graph_constructor.h"
#include "dlxnet/core/graph/graph.h"
#include "dlxnet/core/framework/function.h"
#include "dlxnet/core/framework/session_state.h"

namespace dlxnet{
    class DirectSessionFactory;
    class Device;

    class DirectSession: public Session{
        public:
            DirectSession(const SessionOptions& options, const DeviceMgr* device_mgr,
                    DirectSessionFactory* factory);
            ~DirectSession() override;
            typedef std::vector<std::pair<string, Tensor>> NamedTensorList;

            Status Create(const GraphDef& graph) override;
            Status Create(GraphDef&& graph) override;

            Status Extend(const GraphDef& graph) override;
            Status Extend(GraphDef&& graph) override;

            Status Run(const NamedTensorList& inputs,
                    const std::vector<string>& output_names,
                    const std::vector<string>& target_nodes,
                    std::vector<Tensor>* outputs) override;

            // NOTE: Experimental and subject to change.
            Status Run(const RunOptions& run_options,
                    const NamedTensorList& inputs,
                    const std::vector<string>& output_names,
                    const std::vector<string>& target_nodes,
                    std::vector<Tensor>* outputs,
                    RunMetadata* run_metadata) override;
            Status Run(
                    const RunOptions& run_options,
                    const NamedTensorList& inputs, const std::vector<string>& output_names,
                    const std::vector<string>& target_nodes, std::vector<Tensor>* outputs,
                    RunMetadata* run_metadata,
                    const thread::ThreadPoolOptions& threadpool_options) override;

            // Reset clears 'containers' from the device_mgr of the DirectSession.
            // If 'containers' is empty, then Reset clears the default container.
            Status Reset(const std::vector<string>& containers);

            Status ListDevices(
                    std::vector<DeviceAttributes>* response) override;
            Status Close() override;
            Status LocalDeviceManager(const DeviceMgr** output) override {
                *output = device_mgr_.get();
                return Status::OK();
            }
            const SessionOptions& options() const { return options_; }

        private:
            struct PerPartitionExecutorsAndLib{
                Device* device=nullptr;
                std::unique_ptr<Graph> graph=nullptr;
                std::unique_ptr<Executor> executor;
            };
            struct ExecutorsAndKeys{
                std::vector<PerPartitionExecutorsAndLib> items;
            };
            // Retrieves an already existing set of executors to run 'inputs' and
            // 'outputs', or creates and caches them for future use.
            Status GetOrCreateExecutors(
                    gtl::ArraySlice<string> inputs, gtl::ArraySlice<string> outputs,
                    gtl::ArraySlice<string> target_nodes,
                    ExecutorsAndKeys** executors_and_keys);

            // Creates a set of executors to run the subgraph defined by
            // `callable_options`.
            Status CreateExecutors(
                    const CallableOptions& callable_options,
                    std::unique_ptr<ExecutorsAndKeys>* out_executors_and_keys);
            // Creates several graphs given the existing graph_def_ and the
            // input feeds and fetches, given 'devices'. The graphs share a common
            // function library 'flib_def'.
            Status CreateGraphs(
                    const BuildGraphOptions& options,
                    std::unordered_map<string, std::unique_ptr<Graph>>* outputs,
                    DataTypeVector* input_types, DataTypeVector* output_types,
                    int64* collective_graph_key);

            Status RunInternal(
                    int64 step_id, const RunOptions& run_options,
                    CallFrameInterface* call_frame, ExecutorsAndKeys* executors_and_keys,
                    RunMetadata* run_metadata,
                    const thread::ThreadPoolOptions& threadpool_options);
            Status ExtendLocked(GraphDef graph)
                EXCLUSIVE_LOCKS_REQUIRED(graph_state_lock_);
            Status CheckNotClosed() {
                mutex_lock l(closed_lock_);
                if (closed_) return errors::Cancelled("Session has been closed.");
                return Status::OK();
            }

            Status CheckGraphCreated(const char* method) {
                mutex_lock l(graph_state_lock_);
                if (!graph_created_) {
                    return errors::InvalidArgument(
                            "Session was not created with a graph before ", method, "!");
                }
                return Status::OK();
            }

            const SessionOptions options_;

            // Device structures.
            const std::unique_ptr<const DeviceMgr> device_mgr_;
            std::vector<Device*> devices_;  // not owned
            // Unique session identifier.
            string session_handle_;

            mutex graph_state_lock_;
            bool graph_created_ GUARDED_BY(graph_state_lock_) = false;
            bool finalized_ GUARDED_BY(graph_state_lock_) = false;
            // The thread-pools to use for running ops, with a bool indicating if the pool
            // is owned.
            std::vector<std::pair<thread::ThreadPool*, bool>> thread_pools_;
            Status init_error_;  // Set to an error if construction failed.

            // If true, blocks until device has finished all queued operations in a step.
            bool sync_on_finish_ = true;
            mutex executor_lock_;  // protects executors_
            // Holds mappings from signature to the executors that process
            // it. The reason for a level of indirection around mapped_type is
            // to guarantee address stability.
            // The map value is a shared_ptr since multiple map keys can point to the
            // same ExecutorsAndKey object.
            std::unordered_map<string, std::shared_ptr<ExecutorsAndKeys>> executors_
                GUARDED_BY(executor_lock_);

            // This holds all the tensors that are currently alive in the session.
            SessionState session_state_;
            DirectSessionFactory* const factory_;  // not owned
            // Execution_state; used when placing the entire graph.
            std::unique_ptr<GraphExecutionState> execution_state_
                GUARDED_BY(graph_state_lock_);

            // The function library, before any rewrites or optimizations have been
            // performed. In particular, CreateGraphs() may need to modify the function
            // library; it copies and modifies the function library.
            // std::unique_ptr<FunctionLibraryDefinition> flib_def_;
            // true if the Session has been Closed.
            mutex closed_lock_;
            bool closed_ GUARDED_BY(closed_lock_) = false;
            // For generating unique names for this session instance.
            std::atomic<int64> edge_name_counter_ = {0};
            std::atomic<int64> handle_name_counter_ = {0};
            // Run in caller's thread if RunOptions.inter_op_thread_pool is negative or
            // all of following conditions are met:
            // 1. This session doesn't own any thread pool.
            // 2. RunOptions.inter_op_thread_pool is unspecified or 0.
            // 3. This session has a single executor.
            // 4. config.inter_op_parallelism_threads is specified to negative explicitly
            //    or through environment variable TF_NUM_INTEROP_THREADS.
            // 5. RunOptions.experimental.use_run_handler_pool is unspecified or false.
            // Otherwise run in global thread pool, session owned thread pool or handler
            // pool according to other specifications of RunOptions and ConfigProto.
            bool run_in_caller_thread_ = false;

            TF_DISALLOW_COPY_AND_ASSIGN(DirectSession);

    };
}


#endif
