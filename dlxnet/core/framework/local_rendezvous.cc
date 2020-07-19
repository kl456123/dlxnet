#include "dlxnet/core/framework/local_rendezvous.h"
#include "dlxnet/core/lib/gtl/manual_constructor.h"


namespace dlxnet{
    // Represents a blocked Send() or Recv() call in the rendezvous.
    struct LocalRendezvous::Item {
        enum Type { kSend = 0, kRecv = 1 };

        Item(Rendezvous::Args send_args, const Tensor& value, bool is_dead)
            : Item(send_args, kSend) {
                send_state.value.Init(value);
                send_state.is_dead = is_dead;
            }

        Item(Rendezvous::Args recv_args, Rendezvous::DoneCallback waiter)
            : Item(recv_args, kRecv) {
                recv_state.waiter.Init(std::move(waiter));
                // recv_state.cancellation_token = cancellation_token;
            }

        ~Item() {
            if (args.device_context) {
                args.device_context->Unref();
            }
            if (type == kSend) {
                send_state.value.Destroy();
            } else {
                recv_state.waiter.Destroy();
            }
        }

        const Rendezvous::Args args;
        const Type type;

        // Link to next item in an ItemQueue.
        Item* next = nullptr;

        // The validity of `send_state` or `recv_state` is determined by `type ==
        // kSend` or `type == kRecv` respectively.
        union {
            struct {
                ManualConstructor<Tensor> value;
                bool is_dead;
            } send_state;
            struct {
                ManualConstructor<Rendezvous::DoneCallback> waiter;
                // CancellationToken cancellation_token;
            } recv_state;
        };

        private:
        Item(Rendezvous::Args args, Type type) : args(args), type(type) {
            if (args.device_context) {
                args.device_context->Ref();
            }
        }
    };

    void LocalRendezvous::ItemQueue::push_back(Item* item) {
        if (TF_PREDICT_TRUE(head == nullptr)) {
            // The queue is empty.
            head = item;
            tail = item;
        } else {
            DCHECK_EQ(tail->type, item->type);
            tail->next = item;
            tail = item;
        }
    }

    LocalRendezvous::~LocalRendezvous() {
        if (!table_.empty()) {
            StartAbort(errors::Cancelled("LocalRendezvous deleted"));
        }
    }

    namespace {
        uint64 KeyHash(const StringPiece& k) { return Hash64(k.data(), k.size()); }
    }  // namespace

    Status LocalRendezvous::Send(const Rendezvous::ParsedKey& key,
            const Rendezvous::Args& send_args,
            const Tensor& val, const bool is_dead) {
        uint64 key_hash = KeyHash(key.FullKey());
        DVLOG(2) << "Send " << this << " " << key_hash << " " << key.FullKey();

        mu_.lock();
        if (!status_.ok()) {
            // Rendezvous has been aborted.
            Status s = status_;
            mu_.unlock();
            return s;
        }

        ItemQueue* queue = &table_[key_hash];
        if (queue->head == nullptr || queue->head->type == Item::kSend) {
            // There is no waiter for this message. Append the message
            // into the queue. The waiter will pick it up when arrives.
            // Only send-related fields need to be filled.
            // TODO(b/143786186): Investigate moving the allocation of `Item` outside
            // the lock.
            DVLOG(2) << "Enqueue Send Item (key:" << key.FullKey() << "). ";
            queue->push_back(new Item(send_args, val, is_dead));
            mu_.unlock();
            return Status::OK();
        }

        DVLOG(2) << "Consume Recv Item (key:" << key.FullKey() << "). ";
        // There is an earliest waiter to consume this message.
        Item* item = queue->head;

        // Delete the queue when the last element has been consumed.
        if (item->next == nullptr) {
            DVLOG(2) << "Clean up Send/Recv queue (key:" << key.FullKey() << "). ";
            table_.erase(key_hash);
        } else {
            queue->head = item->next;
        }
        mu_.unlock();

        // Notify the waiter by invoking its done closure, outside the
        // lock.
        DCHECK_EQ(item->type, Item::kRecv);
        (*item->recv_state.waiter)(Status::OK(), send_args, item->args, val, is_dead);
        delete item;
        return Status::OK();
    }

    void LocalRendezvous::RecvAsync(const Rendezvous::ParsedKey& key,
            const Rendezvous::Args& recv_args,
            Rendezvous::DoneCallback done) {
        uint64 key_hash = KeyHash(key.FullKey());
        DVLOG(2) << "Recv " << this << " " << key_hash << " " << key.FullKey();

        mu_.lock();
        if (!status_.ok()) {
            // Rendezvous has been aborted.
            Status s = status_;
            mu_.unlock();
            done(s, Rendezvous::Args(), recv_args, Tensor(), false);
            return;
        }

        ItemQueue* queue = &table_[key_hash];
        if (queue->head == nullptr || queue->head->type == Item::kRecv) {
            DVLOG(2) << "Enqueue Recv Item (key:" << key.FullKey() << "). ";

            queue->push_back(new Item(recv_args, std::move(done)));

            mu_.unlock();
            return;
        }

        DVLOG(2) << "Consume Send Item (key:" << key.FullKey() << "). ";
        // A message has already arrived and is queued in the table under
        // this key.  Consumes the message and invokes the done closure.
        Item* item = queue->head;

        // Delete the queue when the last element has been consumed.
        if (item->next == nullptr) {
            DVLOG(2) << "Clean up Send/Recv queue (key:" << key.FullKey() << "). ";
            table_.erase(key_hash);
        } else {
            queue->head = item->next;
        }
        mu_.unlock();

        // Invoke done() without holding the table lock.
        DCHECK_EQ(item->type, Item::kSend);
        done(Status::OK(), item->args, recv_args, *item->send_state.value,
                item->send_state.is_dead);
        delete item;
    }

    void LocalRendezvous::StartAbort(const Status& status) {
        CHECK(!status.ok());
        Table table;
        {
            mutex_lock l(mu_);
            status_.Update(status);
            table_.swap(table);
        }
        for (auto& p : table) {
            Item* item = p.second.head;
            while (item != nullptr) {
                if (item->type == Item::kRecv) {
                    (*item->recv_state.waiter)(status, Rendezvous::Args(),
                            Rendezvous::Args(), Tensor(), false);
                }
                Item* to_delete = item;
                item = item->next;
                delete to_delete;
            }
        }
    }
} // namespace dlxnet
