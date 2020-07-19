#include "dlxnet/core/framework/rendezvous.h"
#include "dlxnet/core/framework/local_rendezvous.h"


namespace dlxnet{
    Rendezvous::ParsedKey& Rendezvous::ParsedKey::operator=(const ParsedKey& b) {
        const char* b_base = b.buf_.data();
        buf_ = b.buf_;
        src_device = StringPiece(buf_.data() + (b.src_device.data() - b_base),
                b.src_device.size());
        src = b.src;
        src_incarnation = b.src_incarnation;
        dst_device = StringPiece(buf_.data() + (b.dst_device.data() - b_base),
                b.dst_device.size());
        dst = b.dst;
        edge_name = StringPiece(buf_.data() + (b.edge_name.data() - b_base),
                b.edge_name.size());
        return *this;
    }

    /*  static */
    string Rendezvous::CreateKey(const string& src_device, uint64 src_incarnation,
            const string& dst_device, const string& name,
            const FrameAndIter& frame_iter) {
        // NOTE: ';' is not used in the device name's job name.
        //
        // We include both sender and receiver in the key to facilitate
        // debugging. For correctness, we only need to encode the receiver.
        //
        // "src_incarnation" is used to distinguish a worker when it
        // restarts.
        char buf[strings::kFastToBufferSize];
        return strings::StrCat(
                src_device, ";", strings::Uint64ToHexString(src_incarnation, buf), ";",
                dst_device, ";", name, ";", frame_iter.frame_id, ":", frame_iter.iter_id);
    }

    // Return the prefix of "*s" up to the next occurrence of "delim", or
    // the whole remaining string if "delim" is not found.  "*s" is advanced
    // past the string returned plus the delimiter (if found).
    static StringPiece ConsumeNextPart(StringPiece* s, char delim) {
        for (size_t offset = 0; offset < s->size(); offset++) {
            if ((*s)[offset] == delim) {
                StringPiece result(s->data(), offset);
                s->remove_prefix(offset + 1);  // +1: remove delim, as well
                return result;
            }
        }
        // No delimiter found: return rest of string
        StringPiece result(s->data(), s->size());
        s->remove_prefix(s->size());
        return result;
    }

    /* static */
    Status Rendezvous::ParseKey(StringPiece key, ParsedKey* out) {
        if (key.data() == out->buf_.data()) {
            // Caller used our buf_ string directly, so we don't need to copy.  (The
            // SendOp and RecvOp implementations do this, for example).
            DCHECK_EQ(key.size(), out->buf_.size());
        } else {
            // Make a copy that our StringPieces can point at a copy that will persist
            // for the lifetime of the ParsedKey object.
            out->buf_.assign(key.data(), key.size());
        }
        StringPiece s(out->buf_);
        StringPiece parts[5];
        for (int i = 0; i < 5; i++) {
            parts[i] = ConsumeNextPart(&s, ';');
        }
        if (s.empty() &&          // Consumed the whole string
                !parts[4].empty() &&  // Exactly five parts
                DeviceNameUtils::ParseFullName(parts[0], &out->src) &&
                strings::HexStringToUint64(parts[1], &out->src_incarnation) &&
                DeviceNameUtils::ParseFullName(parts[2], &out->dst) &&
                !parts[3].empty()) {
            out->src_device = StringPiece(parts[0].data(), parts[0].size());
            out->dst_device = StringPiece(parts[2].data(), parts[2].size());
            out->edge_name = StringPiece(parts[3].data(), parts[3].size());
            return Status::OK();
        }
        return errors::InvalidArgument("Invalid  rendezvous key: ", key);
    }

    RendezvousInterface::~RendezvousInterface() {}

    namespace {
        class LocalRendezvousWrapper : public Rendezvous {
            public:
                LocalRendezvousWrapper() = default;

                Status Send(const ParsedKey& key, const Args& send_args, const Tensor& val,
                        const bool is_dead) override {
                    return impl_.Send(key, send_args, val, is_dead);
                }

                void RecvAsync(const ParsedKey& key, const Args& recv_args,
                        DoneCallback done) override {
                    impl_.RecvAsync(key, recv_args, std::move(done));
                }

                void StartAbort(const Status& status) override { impl_.StartAbort(status); }

            private:
                LocalRendezvous impl_;

                TF_DISALLOW_COPY_AND_ASSIGN(LocalRendezvousWrapper);
        };
    }  // namespace

    Rendezvous* NewLocalRendezvous() { return new LocalRendezvousWrapper; }
} // namespace dlxnet
