#include "dlxnet/core/public/session.h"



namespace dlxnet{
    Session* NewSession(const SessionOptions& options){
    }

    Status NewSession(const SessionOptions& options, Session** out_session) {
        return Status::OK();
    }
}
