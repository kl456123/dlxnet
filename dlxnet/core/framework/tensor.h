#ifndef MEMORY_MANAGER_CORE_TENSOR_H_
#define MEMORY_MANAGER_CORE_TENSOR_H_
#include <cstdint>
#include <cstddef>

#include "dlxnet/core/framework/tensor_shape.h"
#include "dlxnet/core/framework/allocator.h"
#include "dlxnet/core/framework/types.pb.h"
#include "dlxnet/core/platform/macros.h"
#include "dlxnet/core/framework/tensor_description.pb.h"
#include "dlxnet/core/framework/tensor_types.h"
#include "dlxnet/core/lib/gtl/array_slice.h"
#include "dlxnet/core/framework/types.h"

namespace dlxnet{

    class AllocationDescription;
    class Allocator;
    class Tensor;
    class TensorBuffer;
    class TensorProto;

    class TensorBuffer {
        public:
            explicit TensorBuffer(void* data_ptr) : data_(data_ptr) {}
            virtual ~TensorBuffer() {}

            /// \brief data() points to a memory region of size() bytes.
            ///
            /// NOTE(mrry): The `data()` method is not virtual for performance reasons.
            /// It can be called multiple times when the contents of a `Tensor` are
            /// accessed, and so making it non-virtual allows the body to be inlined.
            void* data() const { return data_; }

            /// \brief Size (in bytes) of the buffer.
            virtual size_t size() const = 0;

            /// \brief Fills metadata about the allocation into the proto.
            virtual void FillAllocationDescription(
                    AllocationDescription* proto) const = 0;

            virtual bool GetAllocatedBytes(size_t* out_bytes) const;

            /// \brief Helper method to reinterpret the buffer as an array of `T`.
            template <typename T>
                T* base() const {
                    return reinterpret_cast<T*>(data());
                }

            /// \brief Whether this TensorBuffer owns the underlying memory.
            virtual bool OwnsMemory() const { return true; }

        private:
            void* const data_;
    };

    class Tensor {
        public:
            /// \brief Creates a 1-dimensional, 0-element float tensor.
            ///
            /// The returned Tensor is not a scalar (shape {}), but is instead
            /// an empty one-dimensional Tensor (shape {0}, NumElements() ==
            /// 0). Since it has no elements, it does not need to be assigned a
            /// value and is initialized by default (IsInitialized() is
            /// true). If this is undesirable, consider creating a one-element
            /// scalar which does require initialization:
            ///
            /// ```c++
            ///
            ///     Tensor(DT_FLOAT, TensorShape({}))
            ///
            /// ```
            Tensor();

            /// \brief Creates a Tensor of the given `type` and `shape`.  If
            /// LogMemory::IsEnabled() the allocation is logged as coming from
            /// an unknown kernel and step. Calling the Tensor constructor
            /// directly from within an Op is deprecated: use the
            /// OpKernelConstruction/OpKernelContext allocate_* methods to
            /// allocate a new tensor, which record the kernel and step.
            ///
            /// The underlying buffer is allocated using a `CPUAllocator`.
            Tensor(DataType type, const TensorShape& shape);

            /// \brief Creates a tensor with the input `type` and `shape`, using
            /// the allocator `a` to allocate the underlying buffer. If
            /// LogMemory::IsEnabled() the allocation is logged as coming from
            /// an unknown kernel and step. Calling the Tensor constructor
            /// directly from within an Op is deprecated: use the
            /// OpKernelConstruction/OpKernelContext allocate_* methods to
            /// allocate a new tensor, which record the kernel and step.
            ///
            /// `a` must outlive the lifetime of this Tensor.
            Tensor(Allocator* a, DataType type, const TensorShape& shape);

            /// \brief Creates a tensor with the input `type` and `shape`, using
            /// the allocator `a` and the specified "allocation_attr" to
            /// allocate the underlying buffer. If the kernel and step are known
            /// allocation_attr.allocation_will_be_logged should be set to true
            /// and LogMemory::RecordTensorAllocation should be called after the
            /// tensor is constructed. Calling the Tensor constructor directly
            /// from within an Op is deprecated: use the
            /// OpKernelConstruction/OpKernelContext allocate_* methods to
            /// allocate a new tensor, which record the kernel and step.
            ///
            /// `a` must outlive the lifetime of this Tensor.
            Tensor(Allocator* a, DataType type, const TensorShape& shape,
                    const AllocationAttributes& allocation_attr);

            /// \brief Creates a tensor with the input datatype, shape and buf.
            ///
            /// Acquires a ref on buf that belongs to this Tensor.
            Tensor(DataType type, const TensorShape& shape, TensorBuffer* buf);

            /// \brief Creates an empty Tensor of the given data type.
            ///
            /// Like Tensor(), returns a 1-dimensional, 0-element Tensor with
            /// IsInitialized() returning True. See the Tensor() documentation
            /// for details.
            explicit Tensor(DataType type);
        public:
            ~Tensor();

            /// Returns the data type.
            DataType dtype() const { return shape_.data_type(); }

            /// Returns the shape of the tensor.
            const TensorShape& shape() const { return shape_; }

            /// \brief Convenience accessor for the tensor shape.
            ///
            /// For all shape accessors, see comments for relevant methods of
            /// `TensorShape` in `tensor_shape.h`.
            int dims() const { return shape().dims(); }

            /// Convenience accessor for the tensor shape.
            int64_t dim_size(int d) const { return shape().dim_size(d); }

            /// Convenience accessor for the tensor shape.
            int64_t NumElements() const { return shape().num_elements(); }

            bool IsSameSize(const Tensor& b) const {
                return shape().IsSameSize(b.shape());
            }
            bool IsInitialized() const;

            /// Returns the estimated memory usage of this tensor.
            size_t TotalBytes() const;

            // Returns the size of allocated memory for this tensor.
            size_t AllocatedBytes() const;
            bool FromProto(const TensorProto& other) TF_MUST_USE_RESULT;
            bool FromProto(Allocator* a, const TensorProto& other) TF_MUST_USE_RESULT;
            void AsProtoField(TensorProto* proto) const;
            void AsProtoTensorContent(TensorProto* proto)const;
            string DebugString(int num_values) const;
            // string DebugString() const;
            string DeviceSafeDebugString() const;
            string DebugString() const { return DebugString(3); }

            StringPiece tensor_data() const;

            void FillDescription(TensorDescription* description) const;
            template<typename T>
                typename TTypes<T>::Flat flat(){
                    // only single dim
                    return shaped<T, 1>({NumElements()});
                }

            template<typename T, size_t NDIMS>
                typename TTypes<T, NDIMS>::Tensor shaped(gtl::ArraySlice<int64> new_sizes);

            // check shape when view tensor(call flat<>() or tensor<>()) bit_casted capable
            template<typename T, size_t NDIMS>
                void FillDimsAndValidateCompatibleShape(gtl::ArraySlice<int64> new_sizes,
                        Eigen::array<Eigen::DenseIndex, NDIMS>* dims);

            template<size_t NDIMS>
                void FillDimsAndValidateCompatibleShape(gtl::ArraySlice<int64> new_sizes,
                        Eigen::array<Eigen::DenseIndex, NDIMS>* dims);
        private:
            void set_dtype(DataType t) { shape_.set_data_type(t); }
            void set_shape(const TensorShape& shape) {
                DataType dt = dtype();
                shape_ = shape;
                set_dtype(dt);
            }
            TensorShape shape_;
            TensorBuffer* buf_;
            // use shaped instead of base to get internal data
            template <typename T>
                T* base() const;
    };

    template <typename T>
        T* Tensor::base() const {
            return buf_ == nullptr ? nullptr : buf_->base<T>();
        }
    template<typename T, size_t NDIMS>
        typename TTypes<T, NDIMS>::Tensor Tensor::shaped(gtl::ArraySlice<int64> new_sizes){
            Eigen::array<Eigen::DenseIndex, NDIMS> dims;
            FillDimsAndValidateCompatibleShape(new_sizes, &dims);
            return typename TTypes<T, NDIMS>::Tensor(base<T>(), dims);
        }

    template<typename T, size_t NDIMS>
        void Tensor::FillDimsAndValidateCompatibleShape(gtl::ArraySlice<int64> new_sizes,
                Eigen::array<Eigen::DenseIndex, NDIMS>* dims){
            // check rank
            CHECK_EQ(NDIMS, new_sizes.size());
            int64 new_num_elements = 1;
            for(size_t i=0;i<NDIMS;i++){
                new_num_elements*= new_sizes[i];
                // fill dims at the same time
                (*dims)[i] = new_sizes[i];
            }
            // check dtype by using memory size consistence
            const int element_size = DataTypeSize(BaseType(dtype()));
            CHECK_EQ(new_num_elements*sizeof(T), NumElements()*element_size);
        }

    template<size_t NDIMS>
        void Tensor::FillDimsAndValidateCompatibleShape(gtl::ArraySlice<int64> new_sizes,
                Eigen::array<Eigen::DenseIndex, NDIMS>* dims){
            // check rank
            CHECK_EQ(NDIMS, new_sizes.size());
            int64 new_num_elements = 1;
            for(size_t i=0;i<NDIMS;i++){
                new_num_elements*= new_sizes[i];
                // fill dims at the same time
                (*dims)[i] = new_sizes[i];
            }
            // check total size
            CHECK_EQ(new_num_elements, NumElements());
        }


}

#endif
