#include "dlxnet/core/public/session.h"

#include <string>

#include "dlxnet/core/common_runtime/session_factory.h"
#include "dlxnet/core/lib/errors.h"
#include "dlxnet/core/platform/logging.h"



namespace dlxnet{
    Session::Session() {}

    Session::~Session() {}

    Status Session::Run(const RunOptions& run_options,
            const std::vector<std::pair<string, Tensor> >& inputs,
            const std::vector<string>& output_tensor_names,
            const std::vector<string>& target_node_names,
            std::vector<Tensor>* outputs, RunMetadata* run_metadata) {
        return errors::Unimplemented(
                "Run with options is not supported for this session.");
    }

    Status Session::PRunSetup(const std::vector<string>& input_names,
            const std::vector<string>& output_names,
            const std::vector<string>& target_nodes,
            string* handle) {
        return errors::Unimplemented(
                "Partial run is not supported for this session.");
    }

    Status Session::PRun(const string& handle,
            const std::vector<std::pair<string, Tensor> >& inputs,
            const std::vector<string>& output_names,
            std::vector<Tensor>* outputs) {
        return errors::Unimplemented(
                "Partial run is not supported for this session.");
    }

    Session* NewSession(const SessionOptions& options){
        Session* out_session;
        Status s = NewSession(options, &out_session);
        if (!s.ok()) {
            LOG(ERROR) << "Failed to create session: " << s;
            return nullptr;
        }
        return out_session;
    }

    Status NewSession(const SessionOptions& options, Session** out_session) {
        SessionFactory* factory;
        Status s = SessionFactory::GetFactory(options, &factory);
        if(!s.ok()){
            *out_session = nullptr;
            LOG(ERROR)<<s;
            return s;
        }
        s = factory->NewSession(options, out_session);
        if(!s.ok()){
            *out_session = nullptr;
        }
        return s;
    }

    Status Reset(const SessionOptions& options,
            const std::vector<string>& containers) {
        SessionFactory* factory;
        TF_RETURN_IF_ERROR(SessionFactory::GetFactory(options, &factory));
        return factory->Reset(options, containers);
    }
}
