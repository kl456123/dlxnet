#include "dlxnet/core/framework/tensor.h"
#include "dlxnet/core/framework/typed_allocator.h"
#include "dlxnet/core/framework/types.h"
#include "dlxnet/core/framework/type_traits.h"
#include "dlxnet/core/framework/allocation_description.pb.h"
#include "dlxnet/core/framework/tensor.pb.h"
#include "dlxnet/core/platform/tensor_coding.h"
#include "dlxnet/core/platform/protobuf.h"
#include "dlxnet/core/framework/log_memory.h"

namespace dlxnet{

    bool TensorBuffer::GetAllocatedBytes(size_t* out_bytes) const {
        AllocationDescription allocation_description;
        FillAllocationDescription(&allocation_description);
        if (allocation_description.allocated_bytes() > 0) {
            *out_bytes = allocation_description.allocated_bytes();
            return true;
        } else {
            return false;
        }
    }

    namespace{
        // some inner classes
        // An un-templated base class for Buffer.
        class BufferBase : public TensorBuffer {
            public:
                explicit BufferBase(Allocator* alloc, void* data_ptr)
                    : TensorBuffer(data_ptr), alloc_(alloc) {}

                bool GetAllocatedBytes(size_t* out_bytes) const override {
                    if (alloc_->TracksAllocationSizes()) {
                        *out_bytes = alloc_->AllocatedSize(data());
                        return *out_bytes > 0;
                    } else {
                        return false;
                    }
                }

                void FillAllocationDescription(AllocationDescription* proto) const override {
                    void* data_ptr = data();
                    int64_t rb = size();
                    proto->set_requested_bytes(rb);
                    proto->set_allocator_name(alloc_->Name());
                    proto->set_ptr(reinterpret_cast<uintptr_t>(data_ptr));
                    if (alloc_->TracksAllocationSizes()) {
                        int64_t ab = alloc_->AllocatedSize(data_ptr);
                        proto->set_allocated_bytes(ab);
                        int64_t id = alloc_->AllocationId(data_ptr);
                        if (id > 0) {
                            proto->set_allocation_id(id);
                        }
                    }
                }

            protected:

                Allocator* const alloc_;
        };

        // Typed ref-counted buffer: T[n].
        template <typename T>
            class Buffer : public BufferBase {
                public:
                    Buffer(Allocator* a, int64_t n);
                    Buffer(Allocator* a, int64_t n, const AllocationAttributes& allocation_attr);

                    size_t size() const override { return sizeof(T) * elem_; }

                private:
                    int64_t elem_;

                    ~Buffer() override;

                    TF_DISALLOW_COPY_AND_ASSIGN(Buffer);
            };
        void LogUnexpectedSize(int64 actual, int64 expected) {
            LOG(ERROR) << "Input size was " << actual << " and expected " << expected;
        }
        // A set of helper functions depending on T.
        // used for simple type(no need to run ctr and dctr)
        template <typename T>
            struct Helper {
                static_assert(is_simple_type<T>::value, "T is not a simple type");
                typedef protobuf::RepeatedField<T> RepeatedFieldType;

                // Encoder of simple type T to a string.  We do a copy.
                template<typename Destination>
                    static void Encode(TensorBuffer* in, int64 n, Destination* out){
                        DCHECK_EQ(in->size(), sizeof(T)*n);
                        port::AssignRefCounted(StringPiece(in->base<const char>(), in->size()), in, out);
                    }
                template<typename Source>
                    static TensorBuffer* Decode(Allocator* a, const Source& in, int64 n){
                        if (in.size() != sizeof(T) * n) {
                            LogUnexpectedSize(in.size(), sizeof(T) * n);
                            return nullptr;
                        }
                        Buffer<T>* buf = new Buffer<T>(a, n);
                        char* data = buf->template base<char>();
                        if (data == nullptr) {
                            buf->Unref();
                            return nullptr;
                        }
                        port::CopyToArray(in, data);
                        return buf;
                    }
                // Memory usage.
                static int64_t TotalBytes(TensorBuffer* in, int64_t n) {
                    CHECK_EQ(in->size(), sizeof(T) * n);
                    return in->size();
                }
            };
        // specialize for complex objects
        template<>
            struct Helper<tstring>{
                typedef protobuf::RepeatedField<string> RepeatedFieldType;
                template<typename Destination>
                    static void Encode(TensorBuffer* in, int64 n, Destination* out){
                        port::EncodeStringList(in->base<const tstring>(), n, out);
                    }
                template<typename Source>
                    static TensorBuffer* Decode(Allocator* a, const Source& in, int64 n){
                        Buffer<tstring>* buf = new Buffer<tstring>(a, n);
                        tstring* strings = buf->template base<tstring>();
                        if (strings == nullptr || !port::DecodeStringList(in, strings, n)) {
                            buf->Unref();
                            return nullptr;
                        }
                        return buf;
                    }

                // Memory usage.
                static int64_t TotalBytes(TensorBuffer* in, int64_t n) {
                    int64 tot = in->size();
                    DCHECK_EQ(tot, sizeof(tstring) * n);
                    const tstring* p = in->base<const tstring>();
                    for (int i = 0; i < n; ++i, ++p) tot += p->size();
                    return tot;
                }
            };

        template <typename T>
            Buffer<T>::Buffer(Allocator* a, int64_t n)
            : BufferBase(a, TypedAllocator::Allocate<T>(a, n, AllocationAttributes())),
            elem_(n) {}

        template <typename T>
            Buffer<T>::Buffer(Allocator* a, int64_t n,
                    const AllocationAttributes& allocation_attr)
            : BufferBase(a, TypedAllocator::Allocate<T>(a, n, allocation_attr)),
            elem_(n) {}

        template <typename T>
            Buffer<T>::~Buffer() {
                if (data()) {
                    TypedAllocator::Deallocate<T>(alloc_, static_cast<T*>(data()), elem_);
                }
            }

        void RefIfNonNull(core::RefCounted* buf) {
            if (buf) buf->Ref();
        }

        void UnrefIfNonNull(core::RefCounted* buf) {
            if (buf) buf->Unref();
        }
    };// end namespace


    Tensor::Tensor() : Tensor(DT_FLOAT) {}

    Tensor::Tensor(DataType type) : shape_(type), buf_(nullptr) {}

    Tensor::Tensor(DataType type, const TensorShape& shape, TensorBuffer* buf)
        : shape_(shape), buf_(buf) {
            set_dtype(type);
        }

    bool Tensor::IsInitialized() const {
        return (buf_ != nullptr && buf_->data() != nullptr) ||
            shape_.num_elements() == 0;
    }

    Tensor::~Tensor() {}
    // The macro CASES() expands to a switch statement conditioned on
    // TYPE_ENUM. Each case expands the STMTS after a typedef for T.
#define SINGLE_ARG(...) __VA_ARGS__
#define CASE(TYPE, STMTS)             \
    case DataTypeToEnum<TYPE>::value: { \
                                          typedef TYPE T;                   \
                                          STMTS;                            \
                                          break;                            \
                                      }
#define CASES_WITH_DEFAULT(TYPE_ENUM, STMTS, INVALID, DEFAULT) \
    switch (TYPE_ENUM) {                                         \
        CASE(float, SINGLE_ARG(STMTS))                             \
        CASE(double, SINGLE_ARG(STMTS))                            \
        CASE(int32_t, SINGLE_ARG(STMTS))                           \
        case DT_INVALID:                                           \
                                                                   INVALID;                                                 \
        break;                                                   \
        default:                                                   \
                                                                   DEFAULT;                                                 \
        break;                                                   \
    }

#define CASES(TYPE_ENUM, STMTS)                                      \
    CASES_WITH_DEFAULT(TYPE_ENUM, STMTS, LOG(FATAL) << "Type not set"; \
            , LOG(FATAL) << "Unexpected type: " << TYPE_ENUM;)


    Tensor::Tensor(Allocator* a, DataType type, const TensorShape& shape)
        : shape_(shape), buf_(nullptr) {
            set_dtype(type);
            // CHECK_NOTNULL(a);
            if (shape_.num_elements() > 0 || a->AllocatesOpaqueHandle()) {
                CASES(type, buf_ = new Buffer<T>(a, shape.num_elements()));
            }
        }

    Tensor::Tensor(Allocator* a, DataType type, const TensorShape& shape,
            const AllocationAttributes& allocation_attr)
        : shape_(shape), buf_(nullptr) {
            set_dtype(type);
            // CHECK_NOTNULL(a);
            if (shape_.num_elements() > 0 || a->AllocatesOpaqueHandle()) {
                CASES(type, buf_ = new Buffer<T>(a, shape.num_elements(), allocation_attr));
            }
        }

    static Allocator* get_default_cpu_allocator() {
        static Allocator* default_cpu_allocator =
            cpu_allocator(port::kNUMANoAffinity);
        return default_cpu_allocator;
    }

    Tensor::Tensor(DataType type, const TensorShape& shape)
        : Tensor(get_default_cpu_allocator(), type, shape) {}


    size_t Tensor::TotalBytes() const {
        if (shape_.num_elements() == 0) return 0;
        CHECK(buf_) << "null buf_ with non-zero shape size " << shape_.num_elements();
        CASES(dtype(), return Helper<T>::TotalBytes(buf_, shape_.num_elements()));
        return 0;  // Makes compiler happy.
    }

    size_t Tensor::AllocatedBytes() const {
        if (buf_) {
            size_t ret;
            if (buf_->GetAllocatedBytes(&ret)) {
                return ret;
            }
        }
        return TotalBytes();
    }

    void Tensor::FillDescription(TensorDescription* description)const{
        description->set_dtype(dtype());
        shape().AsProto(description->mutable_shape());
        if(buf_!=nullptr && buf_->data()!=nullptr){
            buf_->FillAllocationDescription(description->mutable_allocation_description());
        }
    }

    template<typename T>
        struct ProtoHelper{};
    // For a C++ type "T" (float, double, int32, etc.), the repeated field
    // "N"_val (float_val, int_val, label_val, etc.) of type "F" (float,
    // int32, string, etc) in the TensorProto is used for serializing the
    // tensor of type "T".
#define PROTO_TRAITS(T, F, N)                                          \
    template <>                                                          \
    struct ProtoHelper<T> {                                              \
        typedef Helper<F>::RepeatedFieldType FieldType;                    \
        static FieldType::const_iterator Begin(const TensorProto& proto) { \
            return proto.N##_val().begin();                                  \
        }                                                                  \
        static size_t NumElements(const TensorProto& proto) {              \
            return proto.N##_val().size();                                   \
        }                                                                  \
        static void Fill(const T* data, size_t n, TensorProto* proto) {    \
            typename ProtoHelper<T>::FieldType copy(data, data + n);         \
            proto->mutable_##N##_val()->Swap(&copy);                         \
        }                                                                  \
    };
    PROTO_TRAITS(float, float, float);
    PROTO_TRAITS(double, double, double);
    PROTO_TRAITS(int32, int32, int);
    PROTO_TRAITS(uint8, int32, int);
    PROTO_TRAITS(uint16, int32, int);
    PROTO_TRAITS(uint32, uint32, uint32);
    PROTO_TRAITS(int16, int32, int);
    PROTO_TRAITS(int8, int32, int);
    PROTO_TRAITS(bool, bool, bool);
    // PROTO_TRAITS(tstring, tstring, string);
#undef PROTO_TRAITS

    // Allocates a T[n] buffer. Fills in the buffer with repeated values
    // in "in".  If "in" has less values than "n", fills the rest of T[n]
    // with the last value. If "in" has no values, fills T[n] with the
    // default value for T.
    //
    // This routine is using the typed fields (float_val, etc.) in the
    // tensor proto as opposed to the untyped binary representation
    // (tensor_content). This is used when we expect the TensorProto is
    // used by a client program which may not know how to encode a tensor
    // in the compact binary representation.
    template <typename T>
        TensorBuffer* FromProtoField(Allocator* a, const TensorProto& in, int64 n) {
            CHECK_GT(n, 0);
            Buffer<T>* buf = new Buffer<T>(a, n);
            T* data = buf->template base<T>();
            if (data == nullptr) {
                buf->Unref();
                return nullptr;
            }

            const int64 in_n = ProtoHelper<T>::NumElements(in);
            if (in_n <= 0) {
                std::fill_n(data, n, T());
            } else {
                auto begin = ProtoHelper<T>::Begin(in);
                if (n <= in_n) {
                    std::copy_n(begin, n, data);
                } else {
                    std::copy_n(begin, in_n, data);
                    if (std::is_trivially_copyable<T>::value) {
                        const T last = *(data + in_n - 1);
                        std::fill_n(data + in_n, n - in_n, last);
                    } else {
                        const T& last = *(data + in_n - 1);
                        std::fill_n(data + in_n, n - in_n, last);
                    }
                }
            }

            return buf;
        }

    // Copies T[n] stored in the buffer "in" into the repeated field in
    // "out" corresponding to type T.
    template <typename T>
        void ToProtoField(const TensorBuffer& in, int64 n, TensorProto* out) {
            const T* data = in.base<const T>();
            // NOTE: T may not the same as
            // ProtoHelper<T>::FieldType::value_type.  E.g., T==int16,
            // ProtoHelper<T>::FieldType::value_type==int32.  If performance is
            // critical, we can specialize T=float and do memcpy directly.
            ProtoHelper<T>::Fill(data, n, out);
        }

    bool Tensor::FromProto(const TensorProto& proto){
        return FromProto(get_default_cpu_allocator(), proto);
    }
    bool Tensor::FromProto(Allocator* a, const TensorProto& proto){
        CHECK_NOTNULL(a);
        TensorBuffer* p = nullptr;
        if (!TensorShape::IsValid(proto.tensor_shape())) return false;
        if (proto.dtype() == DT_INVALID) return false;
        TensorShape shape(proto.tensor_shape());
        const int64 N = shape.num_elements();
        if (N > 0 && proto.dtype()) {
            bool dtype_error = false;
            if (!proto.tensor_content().empty()) {
                const auto& content = proto.tensor_content();
                CASES_WITH_DEFAULT(proto.dtype(), p = Helper<T>::Decode(a, content, N),
                        dtype_error = true, dtype_error = true);
            } else {
                CASES_WITH_DEFAULT(proto.dtype(), p = FromProtoField<T>(a, proto, N),
                        dtype_error = true, dtype_error = true);
            }
            if (dtype_error || p == nullptr) return false;
        }
        shape_ = shape;
        set_dtype(proto.dtype());
        UnrefIfNonNull(buf_);
        buf_ = p;
        // TODO(misard) add tracking of which kernels and steps are calling
        // FromProto.
        if (buf_ != nullptr && buf_->data() != nullptr && LogMemory::IsEnabled()) {
            LogMemory::RecordTensorAllocation("Unknown (from Proto)",
                    LogMemory::UNKNOWN_STEP_ID, *this);
        }
        return true;
    }
    void Tensor::AsProtoField(TensorProto* proto) const {
        proto->Clear();
        shape_.AsProto(proto->mutable_tensor_shape());
        proto->set_dtype(dtype());
        if (buf_) {
            CASES(dtype(), ToProtoField<T>(*buf_, shape_.num_elements(), proto));
        }
    }

    void Tensor::AsProtoTensorContent(TensorProto* proto) const {
        proto->Clear();
        proto->set_dtype(dtype());
        shape_.AsProto(proto->mutable_tensor_shape());
        if (buf_) {
            CASES(dtype(), Helper<T>::Encode(buf_, shape_.num_elements(),
                        proto->mutable_tensor_content()));
        }
    }

    string Tensor::DebugString(int num_values) const {
        return strings::StrCat("Tensor<type: ", DataTypeString(dtype()),
                " shape: ", shape().DebugString(),
                " values: ", SummarizeValue(num_values), ">");
    }

    string Tensor::DeviceSafeDebugString() const {
        return strings::StrCat("Tensor<type: ", DataTypeString(dtype()),
                " shape: ", shape().DebugString(), ">");
    }

    StringPiece Tensor::tensor_data() const {
        if (buf_ == nullptr) return StringPiece();  // Don't die for empty tensors
        return StringPiece(static_cast<char*>(buf_->data()), TotalBytes());
    }

    void Tensor::CheckType(DataType expected_dtype) const {
        CHECK_EQ(dtype(), expected_dtype)
            << " " << DataTypeString(expected_dtype) << " expected, got "
            << DataTypeString(dtype());
    }

    void Tensor::CheckTypeAndIsAligned(DataType expected_dtype) const {
        CHECK_EQ(dtype(), expected_dtype)
            << " " << DataTypeString(expected_dtype) << " expected, got "
            << DataTypeString(dtype());
        CHECK(IsAligned()) << "ptr = " << base<void>();
    }

    namespace {
        template <typename T>
            string SummarizeArray(int64 limit, int64 num_elts,
                    const TensorShape& tensor_shape, const char* data) {
                string ret;
                const T* array = reinterpret_cast<const T*>(data);

                const gtl::InlinedVector<int64, 4> shape = tensor_shape.dim_sizes();
                for (int64 i = 0; i < limit; ++i) {
                    if (i > 0) strings::StrAppend(&ret, " ");
                    strings::StrAppend(&ret, array[i]);
                }
                if (num_elts > limit) strings::StrAppend(&ret, "...");
                return ret;
            }
    }
    string Tensor::SummarizeValue(int64 max_entries) const {
        const int64 num_elts = NumElements();
        if (max_entries < 0) {
            max_entries = num_elts;
        }
        size_t limit = std::min(max_entries, num_elts);
        if ((limit > 0) && (buf_ == nullptr)) {
            return strings::StrCat("uninitialized Tensor of ", num_elts,
                    " elements of type ", dtype());
        }
        const char* data = limit > 0 ? tensor_data().data() : nullptr;
        switch(dtype()){
            case DT_FLOAT:
                return SummarizeArray<float>(limit, num_elts, shape_, data);
                break;
            case DT_DOUBLE:
                return SummarizeArray<double>(limit, num_elts, shape_, data);
                break;
            case DT_UINT32:
                return SummarizeArray<uint32>(limit, num_elts, shape_, data);
                break;
            case DT_INT32:
                return SummarizeArray<int32>(limit, num_elts, shape_, data);
                break;
            case DT_UINT8:
            case DT_QUINT8:
                return SummarizeArray<uint8>(limit, num_elts, shape_, data);
                break;
            case DT_UINT64:
                return SummarizeArray<uint64>(limit, num_elts, shape_, data);
                break;
            case DT_INT64:
                return SummarizeArray<int64>(limit, num_elts, shape_, data);
                break;
            default:
                return "";
        }
    }

    bool Tensor::CanUseDMA() const {
        CASES(dtype(), return is_simple_type<T>::value);
        return false;  // Makes compiler happy.
    }



#undef CASES
#undef CASE

}
