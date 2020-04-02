#ifndef DLXNET_CORE_FRAMEWORK_TENSOR_SHAPE_H_
#define DLXNET_CORE_FRAMEWORK_TENSOR_SHAPE_H_
#include <cstddef>
#include <cstdint>
#include <limits>
#include <array>
#include <vector>

#include "dlxnet/core/platform/logging.h"
#include "dlxnet/core/platform/errors.h"
#include "dlxnet/core/platform/status.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "dlxnet/core/framework/tensor_shape.pb.h"
#include "dlxnet/core/framework/types.pb.h"
#include "dlxnet/core/lib/gtl/array_slice.h"
#include "dlxnet/core/lib/gtl/inlined_vector.h"

namespace dlxnet{
    template<typename Shape>
        class TensorShapeIter;

    class TensorShapeRep{
        public:
            ~TensorShapeRep();

            /// Copy the specified shape
            TensorShapeRep(const TensorShapeRep& b);
            void operator=(const TensorShapeRep& b);

            /// Move the specified shape.  After moving, `b` is safe for destruction and
            // can be reassigned into, but its dimensions and number of elements can be
            // nonsensical (e.g., negative dimension sizes, or number of elements not
            // properly recomputed).
            TensorShapeRep(TensorShapeRep&& b);
            void operator=(TensorShapeRep&& b);

            /// Clear a tensor shape, producing the scalar shape.
            void Clear();

            // Maximum number of dimensions in a tensor.
            // It's 254 because 255 = kUnknownRank is used to represent unknown rank.
            static constexpr int MaxDimensions() { return 254; }

            /// \brief Returns the number of elements in the tensor.
            ///
            /// We use `int64` and not `size_t` to be compatible with `Eigen::Tensor`
            /// which uses `ptrdiff_t`.  For PartialTensorShape, -1 means not fully
            /// defined.
            int64_t num_elements() const { return num_elements_; }

            /// For error messages.
            std::string DebugString() const;
            static std::string DebugString(const TensorShapeProto& proto);

            void DumpRep() const;  // XXX

        protected:
            // Constructable only via TensorShapeBase
            TensorShapeRep() = default;

            void ClearAllButDataType();

            // We use 16 bytes to represent a TensorShape.  Because we need to
            // be able to support full 64-bit dimension sizes and an arbitrary
            // number of dimensions for a Tensor, but most tensor dimensions are
            // significantly smaller than 64 bits and most tensors are 1, 2, or 3
            // dimensions, we have several representations.
            // Rep16: Supports up to 6 dimensions where each dimension is < 2^16 - 1
            // Rep32: Supports up to 3 dimensions where each dimension is < 2^32 - 1
            // Rep64: Supports arbitrary dimensionality, 64-bit dimensions using
            //        an out of line vector.
            // For PartialTensorShape, a dimension of static_cast<uint??>(-1) is unknown.
            // This value is not allowed in TensorShape either for format compatibility.
            struct Rep16 {
                uint16_t dims_[6];
            };
            struct Rep32 {
                uint32_t dims_[3];
            };
            struct Rep64 {
                std::array<int64_t, 4>* dims_;
            };

            // We use the max value of uint16 or uint32 to represent unknown shapes, so
            // the maximum representable valid shape in these representations is one less.
            static const int64_t kMaxRep16 = std::numeric_limits<uint16_t>::max() - 1;
            static const int64_t kMaxRep32 = std::numeric_limits<uint32_t>::max() - 1;
            static const uint16_t kUnknownRep16 = std::numeric_limits<uint16_t>::max();
            static const uint32_t kUnknownRep32 = std::numeric_limits<uint32_t>::max();

            Rep16* as16() { return reinterpret_cast<Rep16*>(buf()); }
            Rep32* as32() { return reinterpret_cast<Rep32*>(buf()); }
            Rep64* as64() { return reinterpret_cast<Rep64*>(buf()); }

            const Rep16* as16() const { return reinterpret_cast<const Rep16*>(buf()); }
            const Rep32* as32() const { return reinterpret_cast<const Rep32*>(buf()); }
            const Rep64* as64() const { return reinterpret_cast<const Rep64*>(buf()); }

            enum RepTag { REP16 = 0, REP32 = 1, REP_OUT_OF_LINE = 2 };

            // Since we have a convenient extra byte available, we allow the
            // Tensor class to store an 8-bit value in this extra storage.  This
            // allows it to store the Tensor's datatype enum value here and avoid
            // an extra word of storage.
            friend class Tensor;
            friend class TensorShapeTestHelper;
            DataType data_type() const { return static_cast<DataType>(buf()[13]); }
            void set_data_type(DataType dt) {
                // We only have 8 bits available to store DataType, so make sure it fits
                CHECK_LT(static_cast<uint32_t>(dt), 256u);
                buf()[13] = static_cast<uint8_t>(dt);
            }

            // We store the number of dimensions in byte 14, and the RepTag in byte 15.
            // Bytes [0..13] vary depending on the representation.
            // A value of 255 indicates unknown rank in the PartialTensorShape case.
            static const uint8_t kUnknownRank = 255;
            uint8_t ndims_byte() const { return buf()[14]; }
            void set_ndims_byte(uint8_t nd) { buf()[14] = nd; }

            RepTag tag() const { return static_cast<RepTag>(buf()[15]); }
            void set_tag(RepTag tag) { buf()[15] = static_cast<uint8_t>(tag); }

            void set_num_elements(int64_t n) { num_elements_ = n; }

        private:
            void DestructorOutOfLine();
            void SlowCopyFrom(const TensorShapeRep& b);

            uint8_t* buf() { return &u_.buf[0]; }
            const uint8_t* buf() const { return &u_.buf[0]; }

            union {
                uint8_t buf[16];
                // Force data to be aligned enough for a pointer.
                Rep64* unused_aligner;
            } u_;
            int64_t num_elements_;
    };


    // START_SKIP_DOXYGEN
    template <class Shape>
        class TensorShapeIter {
            public:
                TensorShapeIter(const Shape* shape, int d) : shape_(shape), d_(d) {}
                bool operator==(const TensorShapeIter& rhs) {
                    DCHECK(shape_ == rhs.shape_);
                    return d_ == rhs.d_;
                }
                bool operator!=(const TensorShapeIter& rhs) {
                    DCHECK(shape_ == rhs.shape_);
                    return d_ != rhs.d_;
                }
                void operator++() { ++d_; }
                int64 operator*() { return shape_->dim_size(d_); }

            private:
                const Shape* shape_;
                int d_;
        };
    // END_SKIP_DOXYGEN



    /// Base class for TensorShape and PartialTensorShape.
    /// The class is templatized by either TensorShape or PartialTensorShape to
    /// allow skipping known/unknown checks in the TensorShape case, but the
    /// representation is shared exactly for fast conversion.
    template <class Shape>
        class TensorShapeBase : public TensorShapeRep {
            public:
                /// \brief Construct a `TensorShapeBase` from the provided sizes.
                /// REQUIRES: `dim_sizes[i] >= 0` (or >= -1 for PartialTensorShape)
                explicit TensorShapeBase(gtl::ArraySlice<int64> dim_sizes);
                TensorShapeBase(std::initializer_list<int64> dim_sizes)
                    : TensorShapeBase(gtl::ArraySlice<int64>(dim_sizes)) {}

                /// Construct an empty TensorShape, or an unknown rank PartialTensorShape
                TensorShapeBase();

                TensorShapeBase(const TensorShapeProto& proto);

                /// Returns `true` iff `proto` is a valid tensor shape.
                // For TensorShape, the proto shape must be fully defined.
                static bool IsValid(const TensorShapeProto& proto);

                /// Returns `OK` iff `proto` is a valid tensor shape, and a descriptive error
                /// status otherwise.
                static Status IsValidShape(const TensorShapeProto& proto);

                /// Returns `true` iff this is a valid tensor shape.
                bool IsValid();

                /// \brief Add a dimension to the end ("inner-most").
                /// REQUIRES: `size >= 0`
                void AddDim(int64 size);

                /// Appends all the dimensions from `shape`.
                void AppendShape(const TensorShapeBase& shape);

                /// \brief Insert a dimension somewhere in the `TensorShape`.
                /// REQUIRES: `0 <= d <= dims()`
                /// REQUIRES: `size >= 0`
                void InsertDim(int d, int64_t size);

                /// \brief Modifies the size of the dimension `d` to be `size`
                /// REQUIRES: `0 <= d < dims()`
                /// REQUIRES: `size >= 0`
                void set_dim(int d, int64_t size);

                /// \brief Removes dimension `d` from the `TensorShape`.
                /// REQUIRES: `0 <= d < dims()`
                void RemoveDim(int d) {
                    CHECK_GE(d, 0);
                    RemoveDimRange(d, d + 1);
                }

                /// \brief Removes last `n` dimensions from the `TensorShape`.
                /// REQUIRES: `0 <= n <= dims()`
                void RemoveLastDims(int n) {
                    CHECK_LE(n, dims());
                    RemoveDimRange(dims() - n, dims());
                }

                /// \brief Removes the dimensions in range `[begin:end)` from `TensorShape`.
                /// Negative values of `end` are interpreted as `dims() + end + 1` (as in
                /// Python). The same is true for negative values of `begin`. REQUIRES:
                /// `-(dims()+1) <= begin <= dims()` REQUIRES: `-(dims()+1) <= end <= dims()`
                void RemoveDimRange(int begin, int end);

                /// Return whether the rank is unknown
                bool unknown_rank() const {
                    return ndims_byte() == kUnknownRank;
                }

                /// Return the number of dimensions in the tensor.
                /// Can be -1 meaning unknown rank for PartialTensorShape.
                int dims() const {
                    uint8_t dims = ndims_byte();
                    return dims == kUnknownRank ? -1 : dims;
                }

                /// \brief Returns the number of elements in dimension `d`.
                /// REQUIRES: `0 <= d < dims()`
                // TODO(touts): Rename to `dimension()` to match
                // `Eigen::Tensor::dimension()`?
                int64_t dim_size(int d) const;

                /// Returns sizes of all dimensions.
                // Returns an empty list for unknown rank PartialTensorShape.
                gtl::InlinedVector<int64, 4> dim_sizes() const;

                /// Return true iff the rank and all of the dimensions are well defined
                // TODO(irving): Rename to is_fully_defined now that it's fast.
                bool IsFullyDefined() const { return num_elements() != -1; }

                /// Fill `*proto` from `*this`.
                void AsProto(TensorShapeProto* proto) const;

                TensorShapeIter<Shape> begin()const;
                TensorShapeIter<Shape> end()const;


            protected:
                // Optimized constructor for a shape representing an empty vector.
                //
                // This constructor is provided to optimize the default constructor for
                // `Tensor`.
                explicit TensorShapeBase(DataType dt);

            private:
                void RecomputeNumElements();
                void InitDims(gtl::ArraySlice<int64> dim_sizes);

                // Used by AddDim and MakeShapeHelper.  Does no error checking.
                void UnsafeAddDim(int64_t size, int64_t new_num_elements);

        };

    /// Outputs `TensorShapeBase` to `std::ostream`.
    template <typename Shape>
        std::ostream& operator<<(std::ostream& os, const TensorShapeBase<Shape>& tsb) {
            return os << tsb.DebugString();
        }

    /// Represents the shape of a Tensor.
    ///
    /// A tensor's shape is denoted by its number of dimensions and a size for each
    /// dimension.  For example, a Tensor represented by a 3 x 4 matrix would have
    /// a shape of 2-D, [3,4].
    ///
    /// If you know the exact shape of your Tensor when you create the TensorShape
    /// object, you can specify it then, or you can create a TensorShape with
    /// zero dimensions and one element, and call AddDim() to add dimensions later.
    class TensorShape : public TensorShapeBase<TensorShape> {
        public:
            using TensorShapeBase<TensorShape>::TensorShapeBase;

            /// Returns true if `*this` and `b` have the same sizes. Ignores
            /// dimension names.
            bool IsSameSize(const TensorShape& b) const;
            bool operator==(const TensorShape& b) const { return IsSameSize(b); }
            bool operator!=(const TensorShape& b) const { return !IsSameSize(b); }

            /// Fill `*dsizes` from `*this`.
            /// Notice: Using IndexType=int32 in combination with To32Bit() can
            /// significantly improve performance on GPU.
            template <int NDIMS, typename IndexType = Eigen::DenseIndex>
                Eigen::DSizes<IndexType, NDIMS> AsEigenDSizes() const;

            /// Same as `AsEigenDSizes()` but allows for `NDIMS > dims()` -- in
            /// which case we pad the rest of the sizes with 1.
            /// Notice: Using IndexType=int32 in combination with To32Bit() can
            /// significantly improve performance on GPU.
            template <int NDIMS, typename IndexType = Eigen::DenseIndex>
                Eigen::DSizes<IndexType, NDIMS> AsEigenDSizesWithPadding() const;

        private:
            // These CHECK fail to ease debugging.
            // REQUIRES: dims() == NDIMS
            void CheckDimsEqual(int NDIMS) const;
            // REQUIRES: dims() >= NDIMS
            void CheckDimsAtLeast(int NDIMS) const;

            // For access to TensorShapeBase(DataType).
            friend class Tensor;
    };

    /// Represents the value of one dimension in a TensorShape.
    struct TensorShapeDim {
        explicit TensorShapeDim(int64_t s) : size(s) {}
        int64_t size;
    };

    // inline functions
    //
    inline TensorShapeRep::TensorShapeRep(const TensorShapeRep& b) {
        num_elements_ = b.num_elements_;
        if (b.tag() != REP_OUT_OF_LINE) {
            memcpy(buf(), b.buf(), sizeof(u_.buf));
            // memcpy above Implicitly does:
            // set_ndims_byte(b.ndims_byte());
            // set_tag(b.tag());
        } else {
            set_tag(REP16);  // So that SlowCopyFrom does not try to deallocate
            SlowCopyFrom(b);
        }
    }

    inline TensorShapeRep::TensorShapeRep(TensorShapeRep&& b) {
        num_elements_ = b.num_elements_;
        memcpy(buf(), b.buf(), sizeof(u_.buf));
        // memcpy above Implicitly does:
        //   set_ndims_byte(b.ndims_byte());
        //   set_tag(b.tag());
        b.set_tag(REP16);  // other shape no longer owns out-of-line data, if any.
    }

    inline TensorShapeRep::~TensorShapeRep() {
        if (tag() == REP_OUT_OF_LINE) {
            DestructorOutOfLine();
        }
    }

    inline void TensorShapeRep::operator=(const TensorShapeRep& b) {
        num_elements_ = b.num_elements_;
        if (tag() != REP_OUT_OF_LINE && b.tag() != REP_OUT_OF_LINE) {
            memcpy(buf(), b.buf(), sizeof(u_.buf));
            // memcpy above implicitly also does:
            //   set_tag(b.tag());
            //   set_ndims_byte(b.ndims_byte());
        } else {
            SlowCopyFrom(b);
        }
    }

    inline void TensorShapeRep::operator=(TensorShapeRep&& b) {
        if (tag() == REP_OUT_OF_LINE) {
            DestructorOutOfLine();
        }
        num_elements_ = b.num_elements_;
        memcpy(buf(), b.buf(), sizeof(u_.buf));
        // memcpy above Implicitly does:
        //   set_ndims_byte(b.ndims_byte());
        //   set_tag(b.tag());
        b.set_tag(REP16);  // other shape no longer owns out-of-line data, if any.
    }

    template <class Shape>
        inline TensorShapeBase<Shape>::TensorShapeBase(DataType dt) {
            set_tag(REP16);
            set_data_type(dt);

            // Optimized implementation of InitDims() where the shape is statically known
            // to be {0}.
            set_ndims_byte(1);
            uint16_t* dst = as16()->dims_;
            *dst = 0;
            set_num_elements(0);
        }

    template <int NDIMS, typename IndexType>
        Eigen::DSizes<IndexType, NDIMS> TensorShape::AsEigenDSizes() const {
            CheckDimsEqual(NDIMS);
            return AsEigenDSizesWithPadding<NDIMS, IndexType>();
        }

    template <int NDIMS, typename IndexType>
        Eigen::DSizes<IndexType, NDIMS> TensorShape::AsEigenDSizesWithPadding() const {
            CheckDimsAtLeast(NDIMS);
            static_assert(NDIMS <= TensorShape::MaxDimensions(), "Too many dimensions");
            Eigen::DSizes<IndexType, NDIMS> dsizes;
            for (int d = 0; d < dims(); d++) {
                dsizes[d] = static_cast<IndexType>(dim_size(d));
            }
            for (int d = dims(); d < NDIMS; d++) {
                dsizes[d] = 1;
            }
            return dsizes;
        }

    // Declare explicit instantiations in .cc file
    extern template class TensorShapeBase<TensorShape>;

} // namespace dlxnet


#endif
