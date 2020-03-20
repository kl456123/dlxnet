#include "absl/container/flat_hash_set.h"

#include "dlxnet/core/common_runtime/direct_session.h"
#include "dlxnet/core/common_runtime/session_factory.h"
#include "dlxnet/core/common_runtime/device_factory.h"
#include "dlxnet/core/common_runtime/device_mgr.h"
#include "dlxnet/core/lib/random/random.h"
#include "dlxnet/core/framework/logging.h"
#include "dlxnet/core/platform/threadpool_options.h"


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
}
