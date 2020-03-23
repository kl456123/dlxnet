#include "dlxnet/core/framework/tensor_shape.h"
#include "dlxnet/core/framework/types.h"
#include "dlxnet/core/framework/tensor_shape.pb.h"


namespace dlxnet{


    static_assert(sizeof(TensorShapeRep) == sizeof(TensorShape),
            "TensorShape must have no fields beyond TensorShapeRep");

    void TensorShapeRep::ClearAllButDataType() {
        if (tag() == REP_OUT_OF_LINE) {
            delete as64()->dims_;
        }
        set_tag(REP16);
        set_ndims_byte(0);
        // Leaves data_type alone
        set_num_elements(1);
    }

    std::string TensorShapeRep::DebugString() const {
        const auto& shape = *static_cast<const TensorShape*>(this);
        if (shape.unknown_rank()) return "<unknown>";
        std::string s = "[";
        for (int i = 0; i < shape.dims(); i++) {
            if (i > 0) s = s+",";
            int64_t dim = shape.dim_size(i);
            if (dim < 0) {
                s = s + "?";
            } else {
                s = s+ std::to_string(dim);
            }
        }
        s = s+ "]";
        return s;
    }

    void TensorShapeRep::SlowCopyFrom(const TensorShapeRep& b) {
        if (b.tag() != REP_OUT_OF_LINE) {
            if (tag() == REP_OUT_OF_LINE) {
                delete as64()->dims_;
            }
            memcpy(buf(), b.buf(), sizeof(u_.buf));
            // memcpy above implicitly also does:
            //   set_tag(b.tag());
            //   set_ndims_byte(b.ndims_byte());
            //   set_data_type(b.data_type());
        } else {
            CHECK_EQ(b.tag(), REP_OUT_OF_LINE);
            set_ndims_byte(b.ndims_byte());
            set_data_type(b.data_type());
            if (tag() == REP_OUT_OF_LINE) {
                // vector already allocated
                *(as64()->dims_) = *(b.as64()->dims_);
            } else {
                set_tag(REP_OUT_OF_LINE);
                as64()->dims_ = new std::array<int64_t, 4>(*(b.as64()->dims_));
            }
        }
    }
    void TensorShapeRep::DestructorOutOfLine() {
        CHECK(tag() == REP_OUT_OF_LINE);
        delete as64()->dims_;
    }


    template <class Shape>
        static void AppendTo(const TensorShapeBase<Shape>& s,
                std::vector<int64_t>* vals) {
            // for (auto dim : s) {
            // vals->push_back(dim.size);
            // }
        }

    void TensorShape::CheckDimsEqual(int NDIMS) const {
        CHECK_EQ(NDIMS, dims()) << "Asking for tensor of " << NDIMS << " dimensions"
            << " from a tensor of " << dims() << " dimensions";
    }

    void TensorShape::CheckDimsAtLeast(int NDIMS) const {
        CHECK_GE(NDIMS, dims()) << "Asking for tensor of at least " << NDIMS
            << " dimensions from a tensor of " << dims()
            << " dimensions";
    }


    // TODO(slebedev): Consider merging IsValid implementations.
    template <class Shape>
        bool TensorShapeBase<Shape>::IsValid() {
            // NOTE(irving): Unfortunately, TensorShape allows parsing protos with
            // unknown_shape() set, and it seems hard to remove this without backwards
            // compatibility issues.
            return true;
        }

    template <class Shape>
        bool TensorShapeBase<Shape>::IsValid(const TensorShapeProto& proto) {
            // NOTE(irving): Unfortunately, TensorShape allows parsing protos with
            // unknown_shape() set, and it seems hard to remove this without backwards
            // compatibility issues.
            return true;
        }

    template <class Shape>
        void TensorShapeBase<Shape>::InitDims(std::vector<int64_t> dim_sizes) {
            CHECK_EQ(tag(), REP16);

            // Allow sizes that are under kint64max^0.25 so that 4-way multiplication
            // below cannot overflow.
            static const uint64_t kMaxSmall = 0xd744;
            static_assert(kMaxSmall * kMaxSmall * kMaxSmall * kMaxSmall <= kint64max,
                    "bad overflow check");
            bool large_size = false;
            for (auto s : dim_sizes) {
                if (s > kMaxSmall) {
                    large_size = true;
                    break;
                }
            }

            // if (!large_size) {
            // // Every size fits in 16 bits; use fast-paths for dims in {1,2,3,4}.
            // uint16_t* dst = as16()->dims_;
            // switch (dim_sizes.size()) {
            // case 1: {
            // set_ndims_byte(1);
            // const int64_t size = dim_sizes[0];
            // const bool neg = Set16(kIsPartial, dst, 0, size);
            // set_num_elements(neg ? -1 : size);
            // return;
            // }
            // case 2: {
            // set_ndims_byte(2);
            // const int64_t size0 = dim_sizes[0];
            // const int64_t size1 = dim_sizes[1];
            // bool neg = Set16(kIsPartial, dst, 0, size0);
            // neg |= Set16(kIsPartial, dst, 1, size1);
            // set_num_elements(neg ? -1 : (size0 * size1));
            // return;
            // }
            // case 3: {
            // set_ndims_byte(3);
            // const int64 size0 = dim_sizes[0];
            // const int64 size1 = dim_sizes[1];
            // const int64 size2 = dim_sizes[2];
            // bool neg = Set16(kIsPartial, dst, 0, size0);
            // neg |= Set16(kIsPartial, dst, 1, size1);
            // neg |= Set16(kIsPartial, dst, 2, size2);
            // set_num_elements(neg ? -1 : (size0 * size1 * size2));
            // return;
            // }
            // case 4: {
            // set_ndims_byte(4);
            // const int64 size0 = dim_sizes[0];
            // const int64 size1 = dim_sizes[1];
            // const int64 size2 = dim_sizes[2];
            // const int64 size3 = dim_sizes[3];
            // bool neg = Set16(kIsPartial, dst, 0, size0);
            // neg |= Set16(kIsPartial, dst, 1, size1);
            // neg |= Set16(kIsPartial, dst, 2, size2);
            // neg |= Set16(kIsPartial, dst, 3, size3);
            // set_num_elements(neg ? -1 : (size0 * size1 * size2 * size3));
            // return;
            // }
            // }
            // }

            set_ndims_byte(0);
            set_num_elements(1);
            for (int64_t s : dim_sizes) {
                AddDim(s);
            }
        }

    template <class Shape>
        void TensorShapeBase<Shape>::RemoveDimRange(int begin, int end) {
            if (unknown_rank()) return;
            begin = begin < 0 ? dims() + begin + 1 : begin;
            end = end < 0 ? dims() + end + 1 : end;
            CHECK_GE(begin, 0);
            CHECK_LE(begin, dims());
            CHECK_GE(end, 0);
            CHECK_LE(end, dims());
            if (begin >= end) return;
            std::vector<int64_t> vals(8);
            AppendTo(*this, &vals);
            vals.erase(vals.begin() + begin, vals.begin() + end);
            ClearAllButDataType();
            for (auto dval : vals) {
                AddDim(dval);
            }
            RecomputeNumElements();
        }

    template<typename Shape>
        void TensorShapeBase<Shape>::AddDim(int64_t size) {
            if (unknown_rank()) return;
            CHECK_LT(ndims_byte(), MaxDimensions()) << "Too many dimensions in tensor";
            int64_t new_num_elements;
            if (num_elements() < 0 || size < 0) {
                new_num_elements = -1;
            } else {
                new_num_elements = num_elements()* size;
                CHECK_LE(0, new_num_elements);
            }
            UnsafeAddDim(size, new_num_elements);
        }

    template <class Shape>
        void TensorShapeBase<Shape>::UnsafeAddDim(int64_t size, int64_t new_num_elements) {
            const int nd = ndims_byte();
            if (tag() == REP16 && nd < 6 && size < kMaxRep16) {
                as16()->dims_[nd] = size < 0 ? kUnknownRep16 : static_cast<uint16_t>(size);
            } else if (tag() == REP32 && nd < 3 && size < kMaxRep32) {
                as32()->dims_[nd] =
                    size < 0 ? kUnknownRep32 : static_cast<uint32_t>(size);
            } else if (tag() == REP_OUT_OF_LINE) {
                // as64()->dims_->push_back(size);
            } else {
                // Need to change representation
                std::vector<int64_t> vals(8);
                AppendTo(*this, &vals);
                vals.push_back(size);
                // We know we can't be REP16.  See if we have a small enough
                // number of dimensions and each dimension's size is small enough
                // to allow REP32.
                bool can_be_rep32 = (vals.size() <= 3);
                if (can_be_rep32) {
                    for (size_t i = 0; i < vals.size(); i++) {
                        if (vals[i] >= kMaxRep32) {
                            can_be_rep32 = false;
                            break;
                        }
                    }
                }
                if (can_be_rep32) {
                    set_tag(REP32);
                    for (size_t d = 0; d < vals.size(); d++) {
                        as32()->dims_[d] = vals[d] < 0
                            ? kUnknownRep32
                            : static_cast<uint32_t>(vals[d]);
                    }
                } else {
                    set_tag(REP_OUT_OF_LINE);
                    // as64()->dims_ =
                    // new std::array<int64_t, 4>(vals.begin(), vals.end());
                }
            }
            set_ndims_byte(nd + 1);
            set_num_elements(new_num_elements);
        }

    template <class Shape>
        TensorShapeBase<Shape>::TensorShapeBase() {
            set_tag(REP16);
            set_data_type(DT_INVALID);
            set_ndims_byte(0);
            set_num_elements(1);
        }

    template <class Shape>
        TensorShapeBase<Shape>::TensorShapeBase(std::vector<int64_t> dim_sizes) {
            set_tag(REP16);
            set_data_type(DT_INVALID);
            InitDims(dim_sizes);
        }

    template <class Shape>
        void TensorShapeBase<Shape>::RecomputeNumElements() {
            if (unknown_rank()) {
                set_num_elements(-1);
                return;
            }
            int64_t n = 1;
            // for (auto dim : *this) {
            // if (dim.size < 0) {
            // n = -1;
            // break;
            // }
            // n = n* dim.size;
            // CHECK_LE(0, n);
            // }
            set_num_elements(n);
        }

    template <class Shape>
        int64_t TensorShapeBase<Shape>::dim_size(int d) const {
            if (unknown_rank()) return -1;
            CHECK_GE(d, 0);
            CHECK_LT(d, dims());
            if (tag() == REP16) {
                uint16_t dim = as16()->dims_[d];
                if (dim == kUnknownRep16) return -1;
                return dim;
            } else if (tag() == REP32) {
                uint32_t dim = as32()->dims_[d];
                if (dim == kUnknownRep32) return -1;
                return dim;
            } else {
                return (*as64()->dims_)[d];
            }
        }
    template <class Shape>
        void TensorShapeBase<Shape>::AsProto(TensorShapeProto* proto) const {
            proto->Clear();
            if (unknown_rank()) {
                proto->set_unknown_rank(true);
            } else {
                for (int i = 0; i < dims(); i++) {
                    proto->add_dim()->set_size(dim_size(i));
                }
            }
        }

    template <class Shape>
        TensorShapeBase<Shape>::TensorShapeBase(const TensorShapeProto& proto) {
            set_tag(REP16);
            set_data_type(DT_INVALID);
            // NOTE(irving): Unfortunately, TensorShape allows parsing protos with
            // unknown_shape() set, and it seems hard to remove this without backwards
            // compatibility issues.
            if (proto.unknown_rank()) {
                set_ndims_byte(kUnknownRank);
                set_num_elements(-1);
            } else {
                set_ndims_byte(0);
                set_num_elements(1);
                for (const auto& d : proto.dim()) {
                    AddDim(d.size());
                }
            }
        }
    template<class Shape>
        /*static*/ Status TensorShapeBase<Shape>::IsValidShape(const TensorShapeProto& proto){
            int64 num_elements = 1;
            for(const auto& d: proto.dim()){
                if(d.size()<0){
                    return errors::InvalidArgument("Shape ", DebugString(proto),
                            " is not fully defined");
                }
                num_elements *= d.size();
                if (num_elements < 0) {
                    return errors::InvalidArgument(
                            "Shape ", DebugString(proto),
                            " is too large (more than 2**63 - 1 entries)");
                }
            }
            return Status::OK();
        }


    string TensorShapeRep::DebugString(const TensorShapeProto& proto) {
        string s;
        if (proto.unknown_rank()) {
            strings::StrAppend(&s, "<unknown>");
            if (proto.dim_size() == 0) return s;
        }
        strings::StrAppend(&s, "[");
        bool first = true;
        for (const auto& d : proto.dim()) {
            if (!first) strings::StrAppend(&s, ",");
            if (d.size() == -1) {
                strings::StrAppend(&s, "?");
            } else {
                strings::StrAppend(&s, d.size());
            }
            first = false;
        }
        strings::StrAppend(&s, "]");
        return s;
    }



    template class TensorShapeBase<TensorShape>;

} // namespace dlxnet

