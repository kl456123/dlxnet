#include "absl/container/flat_hash_set.h"

#include "dlxnet/core/common_runtime/direct_session.h"
#include "dlxnet/core/common_runtime/session_factory.h"
#include "dlxnet/core/common_runtime/device_factory.h"
#include "dlxnet/core/common_runtime/device_mgr.h"
#include "dlxnet/core/common_runtime/executor.h"
#include "dlxnet/core/common_runtime/executor_factory.h"
#include "dlxnet/core/lib/random/random.h"
#include "dlxnet/core/lib/gtl/inlined_vector.h"
#include "dlxnet/core/framework/logging.h"
#include "dlxnet/core/framework/function.h"
#include "dlxnet/core/platform/threadpool_options.h"
#include "dlxnet/core/graph/graph_partition.h"
#include "dlxnet/core/graph/graph_constructor.h"
#include "dlxnet/core/common_runtime/graph_optimizer.h"


namespace dlxnet{
    // session factory
    class DirectSessionFactory : public SessionFactory {
        public:
            DirectSessionFactory() {}

            bool AcceptsOptions(const SessionOptions& options) override {
                return options.target.empty();
            }

            Status NewSession(const SessionOptions& options,
                    Session** out_session) override {
                const auto& experimental_config = options.config.experimental();
                if (experimental_config.has_session_metadata()) {
                    if (experimental_config.session_metadata().version() < 0) {
                        return errors::InvalidArgument(
                                "Session version shouldn't be negative: ",
                                experimental_config.session_metadata().DebugString());
                    }
                    const string key = GetMetadataKey(experimental_config.session_metadata());
                    mutex_lock l(sessions_lock_);
                    if (!session_metadata_keys_.insert(key).second) {
                        return errors::InvalidArgument(
                                "A session with the same name and version has already been "
                                "created: ",
                                experimental_config.session_metadata().DebugString());
                    }
                }

                // Must do this before the CPU allocator is created.
                // if (options.config.graph_options().build_cost_model() > 0) {
                // EnableCPUAllocatorFullStats(true);
                // }
                std::vector<std::unique_ptr<Device>> devices;
                TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(
                            options, "/job:localhost/replica:0/task:0", &devices));

                DirectSession* session = new DirectSession(
                        options, new StaticDeviceMgr(std::move(devices)), this);
                {
                    mutex_lock l(sessions_lock_);
                    sessions_.push_back(session);
                }
                *out_session = session;
                return Status::OK();
            }

            Status Reset(const SessionOptions& options,
                    const std::vector<string>& containers) override {
                std::vector<DirectSession*> sessions_to_reset;
                {
                    mutex_lock l(sessions_lock_);
                    // We create a copy to ensure that we don't have a deadlock when
                    // session->Close calls the DirectSessionFactory.Deregister, which
                    // acquires sessions_lock_.
                    std::swap(sessions_to_reset, sessions_);
                }
                Status s;
                for (auto session : sessions_to_reset) {
                    s.Update(session->Reset(containers));
                }
                // TODO(suharshs): Change the Reset behavior of all SessionFactories so that
                // it doesn't close the sessions?
                for (auto session : sessions_to_reset) {
                    s.Update(session->Close());
                }
                return s;
            }

            void Deregister(const DirectSession* session) {
                mutex_lock l(sessions_lock_);
                sessions_.erase(std::remove(sessions_.begin(), sessions_.end(), session),
                        sessions_.end());
                if (session->options().config.experimental().has_session_metadata()) {
                    session_metadata_keys_.erase(GetMetadataKey(
                                session->options().config.experimental().session_metadata()));
                }
            }

        private:
            static string GetMetadataKey(const SessionMetadata& metadata) {
                return absl::StrCat(metadata.name(), "/", metadata.version());
            }

            mutex sessions_lock_;
            std::vector<DirectSession*> sessions_ GUARDED_BY(sessions_lock_);
            absl::flat_hash_set<string> session_metadata_keys_ GUARDED_BY(sessions_lock_);
    };

    class DirectSessionRegistrar {
        public:
            DirectSessionRegistrar() {
                SessionFactory::Register("DIRECT_SESSION", new DirectSessionFactory());
            }
    };
    static DirectSessionRegistrar registrar;
    std::atomic_int_fast64_t   DirectSession::step_id_counter_(1);

    // direct session
    DirectSession::DirectSession(const SessionOptions& options, const DeviceMgr* device_mgr,
            DirectSessionFactory* factory)
        :options_(options),
        device_mgr_(device_mgr),
        factory_(factory){
            // build thread pool
            const int thread_pool_size =
                options_.config.session_inter_op_thread_pool_size();
            for(int i=0;i<thread_pool_size;i++){
            }

            session_handle_ =
                strings::StrCat("direct", strings::FpToString(random::New64()));
            int devices_added = 0;

            if (options.config.log_device_placement()) {
                const string mapping_str = device_mgr_->DeviceMappingString();
                if (mapping_str.empty()) {
                    printf("Device mapping: no known devices.\n");
                } else {
                    printf("Device mapping:\n%s", mapping_str.c_str());
                }
                string msg = strings::StrCat("Device mapping:\n", mapping_str);
                if (!logging::LogToListeners(msg)) {
                    LOG(INFO) << msg;
                }
            }

            for (auto d : device_mgr_->ListDevices()) {
                devices_.push_back(d);
                device_set_.AddDevice(d);
                // d->op_segment()->AddHold(session_handle_);

                // The first device added is special: it is the 'client device' (a
                // CPU device) from which we feed and fetch Tensors.
                if (devices_added == 0) {
                    device_set_.set_client_device(d);
                }
                ++devices_added;
            }

        }

    // several run methods
    Status DirectSession::Run(const NamedTensorList& inputs,
            const std::vector<string>& output_names,
            const std::vector<string>& target_nodes,
            std::vector<Tensor>* outputs) {
        RunMetadata run_metadata;
        return Run(RunOptions(), inputs, output_names, target_nodes, outputs,
                &run_metadata);
    }

    Status DirectSession::Run(const RunOptions& run_options,
            const NamedTensorList& inputs,
            const std::vector<string>& output_names,
            const std::vector<string>& target_nodes,
            std::vector<Tensor>* outputs,
            RunMetadata* run_metadata){
        return Run(RunOptions(), inputs, output_names, target_nodes, outputs,
                run_metadata, thread::ThreadPoolOptions());
    }

    Status DirectSession::Run(
            const RunOptions& run_options,
            const NamedTensorList& inputs, const std::vector<string>& output_names,
            const std::vector<string>& target_nodes, std::vector<Tensor>* outputs,
            RunMetadata* run_metadata,
            const thread::ThreadPoolOptions& threadpool_options){
        // check first
        TF_RETURN_IF_ERROR(CheckNotClosed());
        TF_RETURN_IF_ERROR(CheckGraphCreated("Run"));

        // Extract the inputs names for this run of the session.
        std::vector<string> input_tensor_names;
        input_tensor_names.reserve(inputs.size());
        for (const auto& it : inputs) {
            input_tensor_names.push_back(it.first);
        }

        // create Executors
        ExecutorsAndKeys* executors_and_keys;
        TF_RETURN_IF_ERROR(GetOrCreateExecutors(input_tensor_names, output_names,
                    target_nodes, &executors_and_keys));


        // configure call frame
        FunctionCallFrame call_frame(executors_and_keys->input_types,
                executors_and_keys->output_types);
        gtl::InlinedVector<Tensor, 4> feed_args(inputs.size());
        for (const auto& it : inputs) {
            feed_args[executors_and_keys->input_name_to_index[it.first]] = it.second;
        }

        // set feed args
        const Status s = call_frame.SetArgs(feed_args);
        if (errors::IsInternal(s)) {
            return errors::InvalidArgument(s.error_message());
        } else if (!s.ok()) {
            return s;
        }

        const int64  step_id = step_id_counter_.fetch_add(1);

        TF_RETURN_IF_ERROR(RunInternal(step_id, run_options,
                    &call_frame, executors_and_keys, run_metadata,
                    threadpool_options));

        // get outputs from call frame
        if(outputs){
            std::vector<Tensor> sorted_outputs;
            const Status s = call_frame.ConsumeRetvals(
                    &sorted_outputs, /* allow_dead_tensors = */ false);
            if (errors::IsInternal(s)) {
                return errors::InvalidArgument(s.error_message());
            } else if (!s.ok()) {
                return s;
            }

            outputs->clear();
            size_t output_size = 0;
            outputs->reserve(sorted_outputs.size());
            for (int i = 0; i < output_names.size(); ++i) {
                const string& output_name = output_names[i];
                outputs->emplace_back(std::move(sorted_outputs[
                            executors_and_keys->output_name_to_index[output_name]]));
            }
        }
        return Status::OK();
    }

    Status DirectSession::Create(const GraphDef& graph){
        return Create(GraphDef(graph));
    }
    Status DirectSession::Create(GraphDef&& graph){
        // lock and create when graph is not empty
        TF_RETURN_IF_ERROR(init_error_);
        if (graph.node_size() > 0) {
            mutex_lock l(graph_state_lock_);
            if (graph_created_) {
                return errors::AlreadyExists(
                        "A Graph has already been created for this session.");
            }
            return ExtendLocked(std::move(graph));
        }
        return Status::OK();
        // create original graph(graph_execution_state)
    }

    Status DirectSession::Finalize(){
        // after finalization, graph cannot be changed,
        // and discard execution state
        mutex_lock l(graph_state_lock_);
        if (finalized_) {
            return errors::FailedPrecondition("Session already finalized.");
        }
        if (!graph_created_) {
            return errors::FailedPrecondition("Session not yet created.");
        }
        execution_state_.reset();
        finalized_ = true;
        return Status::OK();
    }

    Status DirectSession::ExtendLocked(GraphDef graph){
        if (finalized_) {
            return errors::FailedPrecondition("Session has been finalized.");
        }
        if(!execution_state_){
            // first create
            GraphExecutionStateOptions options;
            options.device_set = &device_set_;
            options.session_options = &options_;
            options.session_handle = session_handle_;
            TF_RETURN_IF_ERROR(GraphExecutionState::MakeForBaseGraph(
                        std::move(graph), options, &execution_state_));
            graph_created_ = true;
        }
    }

    Status DirectSession::Reset(
            const std::vector<string>& containers) {
        device_mgr_->ClearContainers(containers);
        return Status::OK();
    }

    Status DirectSession::Close(){
        // deregister from factory_
        {
            mutex_lock l(closed_lock_);
            if (closed_) return Status::OK();
            closed_ = true;
        }
        if (factory_ != nullptr) factory_->Deregister(this);
        return Status::OK();
    }

    Status DirectSession::ListDevices(
            std::vector<DeviceAttributes>* response){
        // return attributes
        response->clear();
        response->reserve(devices_.size());
        for (Device* d : devices_) {
            const DeviceAttributes& attrs = d->attributes();
            response->emplace_back(attrs);
        }
        return Status::OK();
    }

    Status DirectSession::Extend(const GraphDef& graph) {
        return Extend(GraphDef(graph));
    }

    Status DirectSession::Extend(GraphDef&& graph) {
        TF_RETURN_IF_ERROR(CheckNotClosed());
        mutex_lock l(graph_state_lock_);
        return ExtendLocked(std::move(graph));
    }

    DirectSession::~DirectSession(){
    }

    Status DirectSession::RunInternal(
            int64 step_id, const RunOptions& run_options,
            CallFrameInterface* call_frame, ExecutorsAndKeys* executors_and_keys,
            RunMetadata* run_metadata,
            const thread::ThreadPoolOptions& threadpool_options){
        const uint64 start_time_usecs = options_.env->NowMicros();
        const size_t num_executors = executors_and_keys->items.size();
        thread::ThreadPool* pool = nullptr;

        if (pool == nullptr) {
            // We allow using the caller thread only when having a single executor
            // specified.
            if (executors_and_keys->items.size() > 1) {
                // disable async execution
                // pool = thread_pools_[0].first;
            } else {
                VLOG(1) << "Executing Session::Run() synchronously!";
            }
        }
        // used to run executor
        Executor::Args args;
        args.step_id = step_id;
        args.call_frame = call_frame;
        args.session_state = &session_state_;
        args.session_handle = session_handle_;
        args.sync_on_finish = sync_on_finish_;

        Status run_status;
        Executor::Args::Runner default_runner = nullptr;
        if (pool == nullptr) {
            default_runner = [](Executor::Args::Closure c) { c(); };
        }

        args.runner = default_runner;
        const auto& item = executors_and_keys->items[0];
        // set_threadpool_args_for_item(item, &args);
        run_status = item.executor->Run(args);
        TF_RETURN_IF_ERROR(run_status);

        bool update_cost_model = false;
        // Build and return the cost model as instructed.
        if (update_cost_model) {
            // auto tune
        }

        auto duration = options_.env->NowMicros() - start_time_usecs;
        return Status::OK();
    }

    Status DirectSession::GetOrCreateExecutors(
            gtl::ArraySlice<string> inputs, gtl::ArraySlice<string> outputs,
            gtl::ArraySlice<string> target_nodes,
            ExecutorsAndKeys** executors_and_keys){
        // We could consider some other signature instead of sorting that
        // preserves the same property to avoid the sort in the future.
        std::vector<string> inputs_sorted(inputs.begin(), inputs.end());
        std::sort(inputs_sorted.begin(), inputs_sorted.end());
        std::vector<string> outputs_sorted(outputs.begin(), outputs.end());
        std::sort(outputs_sorted.begin(), outputs_sorted.end());
        std::vector<string> tn_sorted(target_nodes.begin(), target_nodes.end());
        std::sort(tn_sorted.begin(), tn_sorted.end());

        const string sorted_key = strings::StrCat(
                absl::StrJoin(inputs_sorted, ","), "->",
                absl::StrJoin(outputs_sorted, ","), "/", absl::StrJoin(tn_sorted, ","),
                "/");
        // See if we already have the executors for this run.
        {
            mutex_lock l(executor_lock_);
            auto it = executors_.find(sorted_key);
            if (it != executors_.end()) {
                *executors_and_keys = it->second.get();
                return Status::OK();
            }
        }

        // do some cache
        CallableOptions callable_options;
        // feed input names and output names
        callable_options.mutable_feed()->Reserve(inputs_sorted.size());

        for (const string& input : inputs_sorted) {
            callable_options.add_feed(input);
        }
        callable_options.mutable_fetch()->Reserve(outputs_sorted.size());
        for (const string& output : outputs_sorted) {
            callable_options.add_fetch(output);
        }
        callable_options.mutable_target()->Reserve(tn_sorted.size());
        for (const string& target : tn_sorted) {
            callable_options.add_target(target);
        }

        std::unique_ptr<ExecutorsAndKeys> ek;
        TF_RETURN_IF_ERROR(CreateExecutors(callable_options, &ek));

        // cache it
        // Reacquire the lock, try to insert into the map.
        mutex_lock l(executor_lock_);

        // Another thread may have created the entry before us, in which case we will
        // reuse the already created one.
        auto insert_result = executors_.emplace(
                sorted_key, std::shared_ptr<ExecutorsAndKeys>(std::move(ek)));

        // Insert the value under the original key, so the fast path lookup will work
        // if the user uses the same order of inputs, outputs, and targets again.
        // executors_.emplace(key, insert_result.first->second);
        *executors_and_keys = insert_result.first->second.get();
        return Status::OK();
    }

    Status DirectSession::CreateExecutors(
            const CallableOptions& callable_options,
            std::unique_ptr<ExecutorsAndKeys>* out_executors_and_keys){

        std::unique_ptr<ExecutorsAndKeys> ek(new ExecutorsAndKeys);
        ek->callable_options = callable_options;

        // For regular `Run()`, we use the function calling convention, and so
        // maintain a mapping from input/output names to
        // argument/return-value ordinal index.
        for (int i = 0; i < callable_options.feed().size(); ++i) {
            const string& input = callable_options.feed(i);
            ek->input_name_to_index[input] = i;
        }
        for (int i = 0; i < callable_options.fetch().size(); ++i) {
            const string& output = callable_options.fetch(i);
            ek->output_name_to_index[output] = i;
        }

        // create list of subgraphs
        BuildGraphOptions options;
        options.callable_options = callable_options;
        std::unordered_map<string, std::unique_ptr<Graph>> graphs;
        TF_RETURN_IF_ERROR(CreateGraphs(
                    options, &graphs, &ek->input_types,
                    &ek->output_types, &ek->collective_graph_key));

        ek->items.reserve(graphs.size());
        const auto& optimizer_opts =
            options_.config.graph_options().optimizer_options();

        GraphOptimizer optimizer(optimizer_opts);
        // optimize for each partition
        for(auto iter = graphs.begin(); iter != graphs.end(); ++iter){
            const string& partition_name = iter->first;
            std::unique_ptr<Graph>& partition_graph = iter->second;

            Device* device;
            TF_RETURN_IF_ERROR(device_mgr_->LookupDevice(partition_name, &device));

            ek->items.resize(ek->items.size() + 1);
            auto* item = &(ek->items.back());

            // build executor
            LocalExecutorParams params;
            params.device = device;
            params.create_kernel = [this](Device* device, const NodeDef& ndef,
                    OpKernel** kernel) {
                return CreateNonCachedKernel(device, ndef, kernel);
                // NOTE(mrry): We must not share function kernels (implemented
                // using `CallOp`) between subgraphs, because `CallOp::handle_`
                // is tied to a particular subgraph. Even if the function itself
                // is stateful, the `CallOp` that invokes it is not.
                // if (!OpSegment::ShouldOwnKernel(lib, ndef.op())) {
                // return lib->CreateKernel(ndef, kernel);
                // }
                // auto create_fn = [lib, &ndef](OpKernel** kernel) {
                // return lib->CreateKernel(ndef, kernel);
                // };
                // Kernels created for subgraph nodes need to be cached.  On
                // cache miss, create_fn() is invoked to create a kernel based
                // on the function library here + global op registry.
                // return opseg->FindOrCreate(session_handle_, ndef.name(), kernel,
                // create_fn);
            };
            params.delete_kernel = [](OpKernel* kernel) {
                if (kernel){
                    delete kernel;
                }
            };
            // optimizer.Optimize(options_.env, device, &partition_graph,
            // [>shape_map=<]nullptr);

            item->executor = nullptr;
            item->device = device;
            auto executor_type = options_.config.experimental().executor_type();
            TF_RETURN_IF_ERROR(
                    NewExecutor(executor_type, params, *partition_graph, &item->executor));
        }

        // cache
        for (int i = 0; i < callable_options.feed().size(); ++i) {
            const string& input = callable_options.feed(i);
            ek->input_name_to_index[input] = i;
        }
        for (int i = 0; i < callable_options.fetch().size(); ++i) {
            const string& output = callable_options.fetch(i);
            ek->output_name_to_index[output] = i;
        }

        *out_executors_and_keys = std::move(ek);
        return Status::OK();
    }

    Status DirectSession::CreateGraphs(
            const BuildGraphOptions& subgraph_options,
            std::unordered_map<string, std::unique_ptr<Graph>>* outputs,
            DataTypeVector* input_types, DataTypeVector* output_types,
            int64* collective_graph_key){
        mutex_lock l(graph_state_lock_);
        if (finalized_) {
            return errors::FailedPrecondition("Session has been finalized.");
        }
        std::unique_ptr<ClientGraph> client_graph;
        GraphExecutionState* execution_state = nullptr;
        //determine if place graph(subgraph) again or not
        if (options_.config.graph_options().place_pruned_graph()) {

        }else{
            // just use base graph to run
            execution_state = execution_state_.get();
            TF_RETURN_IF_ERROR(
                    execution_state->BuildGraph(subgraph_options, &client_graph));
        }

        // Partition the graph across devices.
        PartitionOptions popts;
        std::unordered_map<string, GraphDef> partitions;
        TF_RETURN_IF_ERROR(Partition(popts, &client_graph->graph, &partitions));

        std::vector<string> device_names;
        for (auto device : devices_) {
            // Extract the LocalName from the device.
            device_names.push_back(DeviceNameUtils::LocalName(device->name()));
        }
        // Check for valid partitions.
        for (const auto& partition : partitions) {
            const string local_partition_name =
                DeviceNameUtils::LocalName(partition.first);
            if (std::count(device_names.begin(), device_names.end(),
                        local_partition_name) == 0) {
                return errors::InvalidArgument(
                        "Creating a partition for ", local_partition_name,
                        " which doesn't exist in the list of available devices. Available "
                        "devices: ",
                        absl::StrJoin(device_names, ","));
            }
        }

        // for each partition, convert them(graph_def) to graph
        for (auto& partition : partitions) {
            std::unique_ptr<Graph> device_graph(
                    new Graph(OpRegistry::Global()));
            GraphConstructorOptions device_opts;
            // There are internal operations (e.g., send/recv) that we now allow.
            // device_opts.allow_internal_ops = true;
            // device_opts.expect_device_spec = true;
            TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
                        device_opts, std::move(partition.second), device_graph.get()));
            outputs->emplace(partition.first, std::move(device_graph));
        }

        // post partition optimization

        Status s;
        for (auto& partition : *outputs) {
            const string& partition_name = partition.first;
            std::unique_ptr<Graph>* graph = &partition.second;

            // VLOG(2) << "Created " << DebugString(graph->get()) << " for "
            // << partition_name;

            // Give the device an opportunity to rewrite its subgraph.
            Device* d;
            s = device_mgr_->LookupDevice(partition_name, &d);
            if (!s.ok()) break;
        }
        std::swap(*input_types, client_graph->feed_types);
        std::swap(*output_types, client_graph->fetch_types);
        return s;
    }
}
