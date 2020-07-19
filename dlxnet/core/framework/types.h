#ifndef DLXNET_CORE_FRAMEWORK_TYPES_H_
#define DLXNET_CORE_FRAMEWORK_TYPES_H_
#include "dlxnet/core/lib/gtl/array_slice.h"
#include "dlxnet/core/lib/gtl/inlined_vector.h"
#include "dlxnet/core/platform/macros.h"
#include "dlxnet/core/platform/logging.h"
#include "dlxnet/core/platform/types.h"
#include "dlxnet/core/framework/types.pb.h"
#include "dlxnet/core/lib/core/stringpiece.h"


namespace dlxnet{
    // MemoryType is used to describe whether input or output Tensors of
    // an OpKernel should reside in "Host memory" (e.g., CPU memory) or
    // "Device" Memory (CPU memory for CPU devices, GPU memory for GPU
    // devices).
    enum MemoryType {
        DEVICE_MEMORY = 0,
        HOST_MEMORY = 1,
    };

    // A DeviceType is just a string, but we wrap it up in a class to give
    // some type checking as we're passing these around
    class DeviceType {
        public:
            DeviceType(const char* type)  // NOLINT(runtime/explicit)
                : type_(type) {}

            explicit DeviceType(StringPiece type) : type_(type.data(), type.size()) {}

            const char* type() const { return type_.c_str(); }
            const string& type_string() const { return type_; }

            bool operator<(const DeviceType& other) const;
            bool operator==(const DeviceType& other) const;
            bool operator!=(const DeviceType& other) const { return !(*this == other); }

        private:
            string type_;
    };
    std::ostream& operator<<(std::ostream& os, const DeviceType& d);

    // Convenient constants that can be passed to a DeviceType constructor
    TF_EXPORT extern const char* const DEVICE_DEFAULT;  // "DEFAULT"
    TF_EXPORT extern const char* const DEVICE_CPU;      // "CPU"
    TF_EXPORT extern const char* const DEVICE_GPU;      // "GPU"
    TF_EXPORT extern const char* const DEVICE_SYCL;     // "SYCL"

    typedef gtl::InlinedVector<MemoryType, 4> MemoryTypeVector;
    typedef gtl::ArraySlice<MemoryType> MemoryTypeSlice;

    typedef gtl::InlinedVector<DataType, 4> DataTypeVector;
    typedef gtl::ArraySlice<DataType> DataTypeSlice;

    typedef gtl::InlinedVector<DeviceType, 4> DeviceTypeVector;
    typedef gtl::InlinedVector<std::pair<DeviceType, int32>, 4>
        PrioritizedDeviceTypeVector;

    // DT_FLOAT + kDataTypeRefOffset == DT_FLOAT_REF, etc.
    enum {kDataTypeRefOffset=100};
    inline bool IsRefType(DataType dtype){
        return dtype>static_cast<DataType>(kDataTypeRefOffset);
    }

    inline DataType MakeRefType(DataType dtype) {
        DCHECK(!IsRefType(dtype));
        return static_cast<DataType>(dtype + kDataTypeRefOffset);
    }
    inline DataType RemoveRefType(DataType dtype) {
        DCHECK(IsRefType(dtype));
        return static_cast<DataType>(dtype - kDataTypeRefOffset);
    }
    inline DataType BaseType(DataType dtype) {
        return IsRefType(dtype) ? RemoveRefType(dtype) : dtype;
    }
    // Returns true if the actual type is the same as or ref of the expected type.
    inline bool TypesCompatible(DataType expected, DataType actual) {
        return expected == actual || expected == BaseType(actual);
    }

    // Validates type T for whether it is a supported DataType.
    template <class T>
        struct IsValidDataType;

    // DataTypeToEnum<T>::v() and DataTypeToEnum<T>::value are the DataType
    // constants for T, e.g. DataTypeToEnum<float>::v() is DT_FLOAT.
    template <class T>
        struct DataTypeToEnum {
            static_assert(IsValidDataType<T>::value, "Specified Data Type not supported");
        };  // Specializations below

    // EnumToDataType<VALUE>::Type is the type for DataType constant VALUE, e.g.
    // EnumToDataType<DT_FLOAT>::Type is float.
    template <DataType VALUE>
        struct EnumToDataType {};  // Specializations below

    // Template specialization for both DataTypeToEnum and EnumToDataType.
#define MATCH_TYPE_AND_ENUM(TYPE, ENUM)                 \
    template <>                                           \
    struct DataTypeToEnum<TYPE> {                         \
        static DataType v() { return ENUM; }                \
        static DataType ref() { return MakeRefType(ENUM); } \
        static constexpr DataType value = ENUM;             \
    };                                                    \
    template <>                                           \
    struct IsValidDataType<TYPE> {                        \
        static constexpr bool value = true;                 \
    };                                                    \
    template <>                                           \
    struct EnumToDataType<ENUM> {                         \
        typedef TYPE Type;                                  \
    }

    MATCH_TYPE_AND_ENUM(float, DT_FLOAT);
    MATCH_TYPE_AND_ENUM(double, DT_DOUBLE);
    MATCH_TYPE_AND_ENUM(int32, DT_INT32);
    MATCH_TYPE_AND_ENUM(uint32, DT_UINT32);
    MATCH_TYPE_AND_ENUM(uint16, DT_UINT16);
    MATCH_TYPE_AND_ENUM(uint8, DT_UINT8);
    MATCH_TYPE_AND_ENUM(int16, DT_INT16);
    MATCH_TYPE_AND_ENUM(int8, DT_INT8);
    MATCH_TYPE_AND_ENUM(tstring, DT_STRING);
    // MATCH_TYPE_AND_ENUM(complex64, DT_COMPLEX64);
    // MATCH_TYPE_AND_ENUM(complex128, DT_COMPLEX128);
    MATCH_TYPE_AND_ENUM(int64, DT_INT64);
    MATCH_TYPE_AND_ENUM(uint64, DT_UINT64);
    MATCH_TYPE_AND_ENUM(bool, DT_BOOL);
    // MATCH_TYPE_AND_ENUM(qint8, DT_QINT8);
    // MATCH_TYPE_AND_ENUM(quint8, DT_QUINT8);
    // MATCH_TYPE_AND_ENUM(qint16, DT_QINT16);
    // MATCH_TYPE_AND_ENUM(quint16, DT_QUINT16);
    // MATCH_TYPE_AND_ENUM(qint32, DT_QINT32);
    // MATCH_TYPE_AND_ENUM(bfloat16, DT_BFLOAT16);
    // MATCH_TYPE_AND_ENUM(Eigen::half, DT_HALF);
    // MATCH_TYPE_AND_ENUM(ResourceHandle, DT_RESOURCE);
    // MATCH_TYPE_AND_ENUM(Variant, DT_VARIANT);

#undef MATCH_TYPE_AND_ENUM

    // All types not specialized are marked invalid.
    template <class T>
        struct IsValidDataType {
            static constexpr bool value = false;
        };

    // Extra validity checking; not part of public API.
    static_assert(IsValidDataType<int64>::value, "Incorrect impl for int64");
    static_assert(IsValidDataType<int32>::value, "Incorrect impl for int32");

    // DataTypeSet represents a set of DataType values as a simple and efficient
    // bit mask.  Note that DataTypeSet cannot represent all DataType values; it
    // cannot represent any of the DT_*_REF values.
    class DataTypeSet {
        private:
            const uint32 mask_;

            static constexpr uint32 kNumBits = 32;

        public:
            constexpr DataTypeSet(const DataTypeSet& other) : mask_(other.mask_) {}
            explicit constexpr DataTypeSet(uint32 mask) : mask_(mask) {}

            constexpr bool Contains(DataType dt) const {
                return (static_cast<uint32>(dt) < kNumBits) &&
                    ((mask_ >> static_cast<uint32>(dt)) & 1u) != 0u;
            }

            class Iterator {
                const DataTypeSet& set_;
                uint32 pos_;

                public:
                Iterator(const DataTypeSet& set, uint32 pos) : set_(set), pos_(pos) {
                    DCHECK_LE(pos, kNumBits);
                }
                DataType operator*() const { return static_cast<DataType>(pos_); }
                Iterator& operator++() {
                    ++pos_;
                    DCHECK_LE(pos_, kNumBits);
                    if (pos_ < kNumBits) {
                        uint32 remaining_mask = set_.mask_ >> pos_;
                        if (remaining_mask != 0u) {
                            pos_ += ctz_uint32(remaining_mask);
                        }
                    }
                    DCHECK_LE(pos_, kNumBits);
                    return *this;
                }
                bool operator==(const Iterator& other) const { return pos_ == other.pos_; }
                bool operator!=(const Iterator& other) const { return !(*this == other); }
                size_t operator-(const Iterator& other) const {
                    return this->pos_ - other.pos_;
                }
            };

            static uint32 ctz_uint32(uint32 x) {
                DCHECK_NE(x, 0u);
#ifdef __GNUC__
                return __builtin_ctz(x);
#else
                uint32 n = 0u;
                while ((x & 1u) == 0u) {
                    x >>= 1;
                    ++n;
                }
                return n;
#endif
            }

            static uint32 clz_uint32(uint32 x) {
                DCHECK_NE(x, 0u);
#ifdef __GNUC__
                return __builtin_clz(x);
#else
                uint32 n = 0u;
                while ((x >> (kNumBits - 1u)) == 0u) {
                    x <<= 1;
                    ++n;
                }
                return n;
#endif
            }

            Iterator begin() const {
                // The begin position is the index of the first bit set to 1 in the entire
                // bit mask. If there are no bits set to 1, then the index is 0.
                if (mask_ != 0) {
                    return Iterator(*this, ctz_uint32(mask_));
                }
                // The set is empty.
                return Iterator(*this, 0);
            }

            Iterator end() const {
                // The end position is the index of the highest bit that is set, plus 1.
                // If there are no bits set to 1, then the index is 0.
                if (mask_ != 0) {
                    return Iterator(*this, kNumBits - clz_uint32(mask_));
                }
                // The set is empty.
                return Iterator(*this, 0);
            }

            size_t size() const {
#if defined(__GNUC__)
                return __builtin_popcount(mask_);
#else
                size_t n = 0;
                uint32 x = mask_;
                while (x > 0) {
                    n += x & 1u;
                    x >>= 1;
                }
                return n;
#endif
            }

            constexpr DataTypeSet operator|(const DataTypeSet& other) const {
                return DataTypeSet(mask_ | other.mask_);
            }
    };

    constexpr inline DataTypeSet ToSet(DataType dt) {
        return DataTypeSet(1u << static_cast<uint32>(dt));
    }

    // Types that support '<' and '>'.
    constexpr DataTypeSet kRealNumberTypes =
        ToSet(DT_FLOAT) | ToSet(DT_DOUBLE) | ToSet(DT_INT32) | ToSet(DT_INT64) |
        ToSet(DT_UINT8) | ToSet(DT_INT16) | ToSet(DT_INT8) | ToSet(DT_UINT16) |
        ToSet(DT_HALF) | ToSet(DT_UINT32) | ToSet(DT_UINT64) | ToSet(DT_BFLOAT16);
    inline const DataTypeSet RealNumberTypes() { return kRealNumberTypes; }

    // Return the list of all numeric types.
    // Includes complex and quantized types.
    // NOTE: On Android, we only include the float and int32 types for now.
    const DataTypeSet kNumberTypes =
        ToSet(DT_FLOAT) | ToSet(DT_DOUBLE) | ToSet(DT_INT64) | ToSet(DT_INT32) |
        ToSet(DT_UINT8) | ToSet(DT_UINT16) | ToSet(DT_INT16) | ToSet(DT_INT8) |
        ToSet(DT_COMPLEX64) | ToSet(DT_COMPLEX128) | ToSet(DT_QINT8) |
        ToSet(DT_QUINT8) | ToSet(DT_QINT32) | ToSet(DT_HALF) | ToSet(DT_UINT32) |
        ToSet(DT_UINT64) | ToSet(DT_BFLOAT16);
    inline const DataTypeSet& NumberTypes() { return kNumberTypes; }

    constexpr DataTypeSet kQuantizedTypes = ToSet(DT_QINT8) | ToSet(DT_QUINT8) |
        ToSet(DT_QINT16) | ToSet(DT_QUINT16) |
        ToSet(DT_QINT32);
    inline const DataTypeSet& QuantizedTypes() { return kQuantizedTypes; }

    // Types that support '<' and '>', including quantized types.
    const DataTypeSet kRealAndQuantizedTypes =
        ToSet(DT_FLOAT) | ToSet(DT_DOUBLE) | ToSet(DT_INT32) | ToSet(DT_INT64) |
        ToSet(DT_UINT8) | ToSet(DT_UINT16) | ToSet(DT_INT16) | ToSet(DT_INT8) |
        ToSet(DT_QINT8) | ToSet(DT_QUINT8) | ToSet(DT_QINT16) | ToSet(DT_QUINT16) |
        ToSet(DT_QINT32) | ToSet(DT_HALF) | ToSet(DT_BFLOAT16);
    inline const DataTypeSet& RealAndQuantizedTypes() {
        return kRealAndQuantizedTypes;
    }


    bool DataTypeFromString(StringPiece sp, DataType* dt);
    string DataTypeString(DataType dtype);
    string DeviceTypeString(const DeviceType& device_type);

    string DataTypeSliceString(const DataTypeSlice dtypes);
    inline string DataTypeVectorString(const DataTypeVector& dtypes) {
        return DataTypeSliceString(dtypes);
    }

    // Returns a 0 on failure
    int DataTypeSize(DataType dt);

    // Types that always sit on host: DT_STRING, DT_STRING_REF, DT_RESOURCE.
    // For DT_RESOURCE, the handle always sits on host (even if the underlying
    // object has device-allocated resources).
    bool DataTypeAlwaysOnHost(DataType dt);

}

#endif
