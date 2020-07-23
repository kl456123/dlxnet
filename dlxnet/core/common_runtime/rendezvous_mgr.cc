#include "dlxnet/core/common_runtime/rendezvous_mgr.h"

namespace dlxnet{
    RefCountedIntraProcessRendezvous::RefCountedIntraProcessRendezvous(
            const DeviceMgr* device_mgr)
        : device_mgr_(device_mgr) {}

    RefCountedIntraProcessRendezvous::~RefCountedIntraProcessRendezvous() {}

    Status RefCountedIntraProcessRendezvous::Send(const ParsedKey& key,
            const Rendezvous::Args& args,
            const Tensor& val,
            const bool is_dead) {
        VLOG(1) << "IntraProcessRendezvous Send " << this << " " << key.FullKey();
        return local_.Send(key, args, val, is_dead);
    }

    void RefCountedIntraProcessRendezvous::RecvAsync(const ParsedKey& key,
            const Rendezvous::Args& args,
            DoneCallback done) {
        VLOG(1) << "IntraProcessRendezvous Recv " << this << " " << key.FullKey();
        // IntraProcessRecvAsyncImpl(device_mgr_, &local_, key, args, std::move(done));
    }

    void RefCountedIntraProcessRendezvous::StartAbort(const Status& s) {
        local_.StartAbort(s);
    }

    PrivateIntraProcessRendezvous::PrivateIntraProcessRendezvous(
            const DeviceMgr* device_mgr)
        : device_mgr_(device_mgr) {}

    PrivateIntraProcessRendezvous::~PrivateIntraProcessRendezvous() {}

    Status PrivateIntraProcessRendezvous::Send(const ParsedKey& key,
            const Rendezvous::Args& args,
            const Tensor& val,
            const bool is_dead) {
        DVLOG(1) << "IntraProcessRendezvous Send " << this << " " << key.FullKey();
        return local_.Send(key, args, val, is_dead);
    }

    void PrivateIntraProcessRendezvous::RecvAsync(const ParsedKey& key,
            const Rendezvous::Args& args,
            DoneCallback done) {
        DVLOG(1) << "StackAllocatedIntraProcessRendezvous Recv " << this << " "
            << key.FullKey();
        // IntraProcessRecvAsyncImpl(device_mgr_, &local_, key, args, std::move(done));
    }

    void PrivateIntraProcessRendezvous::StartAbort(const Status& s) {
        local_.StartAbort(s);
    }
} // namespace dlxnet
