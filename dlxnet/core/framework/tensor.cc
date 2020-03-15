#include "dlxnet/core/framework/tensor.h"
#include "dlxnet/core/framework/typed_allocator.h"
#include "dlxnet/core/framework/types.h"
#include "dlxnet/core/framework/allocation_description.pb.h"
#include "dlxnet/core/framework/tensor.pb.h"

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
        // A set of helper functions depending on T.
        template <typename T>
            struct Helper {

                // Memory usage.
                static int64_t TotalBytes(TensorBuffer* in, int64_t n) {
                    CHECK_EQ(in->size(), sizeof(T) * n);
                    return in->size();
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

    bool Tensor::FromProto(const TensorProto& proto){
        return FromProto(get_default_cpu_allocator(), proto);
    }
    bool Tensor::FromProto(Allocator* a, const TensorProto& proto){
        // CHECK_NOTNULL(a);
        // TensorBuffer* p = nullptr;
        // if (!TensorShape::IsValid(proto.tensor_shape())) return false;
        // if (proto.dtype() == DT_INVALID) return false;
        // TensorShape shape(proto.tensor_shape());
        // const int64 N = shape.num_elements();
        // if (N > 0 && proto.dtype()) {
            // bool dtype_error = false;
            // if (!proto.tensor_content().empty()) {
                // const auto& content = proto.tensor_content();
                // CASES_WITH_DEFAULT(proto.dtype(), p = Helper<T>::Decode(a, content, N),
                        // dtype_error = true, dtype_error = true);
            // } else {
                // CASES_WITH_DEFAULT(proto.dtype(), p = FromProtoField<T>(a, proto, N),
                        // dtype_error = true, dtype_error = true);
            // }
            // if (dtype_error || p == nullptr) return false;
        // }
        // shape_ = shape;
        // set_dtype(proto.dtype());
        // UnrefIfNonNull(buf_);
        // buf_ = p;
        // // TODO(misard) add tracking of which kernels and steps are calling
        // // FromProto.
        // if (buf_ != nullptr && buf_->data() != nullptr && LogMemory::IsEnabled()) {
            // LogMemory::RecordTensorAllocation("Unknown (from Proto)",
                    // LogMemory::UNKNOWN_STEP_ID, *this);
        // }
        return true;
    }
    void Tensor::AsProtoField(TensorProto* proto) const {
        proto->Clear();
        // shape_.AsProto(proto->mutable_tensor_shape());
        // proto->set_dtype(dtype());
        // if (buf_) {
        // CASES(dtype(), ToProtoField<T>(*buf_, shape_.num_elements(), proto));
        // }
    }

    void Tensor::AsProtoTensorContent(TensorProto* proto) const {
    }

    string Tensor::DebugString(int num_values) const {
        return strings::StrCat("Tensor<type: ", DataTypeString(dtype()),
                " shape: ", shape().DebugString(),
                " values: ", "...", ">");
    }

    string Tensor::DeviceSafeDebugString() const {
        return strings::StrCat("Tensor<type: ", DataTypeString(dtype()),
                " shape: ", shape().DebugString(), ">");
    }



#undef CASES
#undef CASE

}
