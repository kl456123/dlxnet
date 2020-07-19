#ifndef DLXNET_CORE_FRAMEWORK_LOCAL_RENDEZVOUS_H_
#define DLXNET_CORE_FRAMEWORK_LOCAL_RENDEZVOUS_H_
#include "dlxnet/core/framework/tensor.h"
#include "dlxnet/core/framework/rendezvous.h"
#include "dlxnet/core/lib/core/status.h"
#include "dlxnet/core/lib/gtl/flatmap.h"
#include "dlxnet/core/platform/mutex.h"
#include "dlxnet/core/platform/types.h"
#include "dlxnet/core/platform/macros.h"


namespace dlxnet{
    // Implements the basic logic of matching Send and Recv operations. See
    // RendezvousInterface for more details.
    //
    // NOTE: Most users will use a class that wraps LocalRendezvous, such as
    // IntraProcessRendezvous or RemoteRendezvous. This class does not implement
    // RendezvousInterface because virtual dispatch to LocalRendezvous methods
    // is not expected to be needed.
    class LocalRendezvous {
        public:
            LocalRendezvous() = default;
            ~LocalRendezvous();

            Status Send(const Rendezvous::ParsedKey& key,
                    const Rendezvous::Args& send_args, const Tensor& val,
                    const bool is_dead);
            void RecvAsync(const Rendezvous::ParsedKey& key,
                    const Rendezvous::Args& recv_args,
                    Rendezvous::DoneCallback done);
            void StartAbort(const Status& status);

        private:
            struct Item;
            // By invariant, the item queue under each key is of the form
            //   [item.type == kSend]* meaning each item is a sent message.
            // or
            //   [item.type == kRecv]* meaning each item is a waiter.
            struct ItemQueue {
                void push_back(Item* item);

                Item* head = nullptr;
                Item* tail = nullptr;
            };

            typedef gtl::FlatMap<uint64, ItemQueue> Table;

            // TODO(zhifengc): shard table_.
            mutex mu_;
            Table table_ GUARDED_BY(mu_);
            Status status_ GUARDED_BY(mu_);

            TF_DISALLOW_COPY_AND_ASSIGN(LocalRendezvous);
    };
}// namespace dlxnet



#endif
