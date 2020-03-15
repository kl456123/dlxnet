#include "dlxnet/core/common_runtime/direct_session.h"


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
}
