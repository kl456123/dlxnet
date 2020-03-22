#ifndef DLXNET_CORE_FRAMEWORK_SHAPE_INFERENCE_H_
#define DLXNET_CORE_FRAMEWORK_SHAPE_INFERENCE_H_
#include <functional>
#include <vector>

#include "dlxnet/core/lib/status.h"
#include "dlxnet/core/framework/tensor_shape.h"
#include "dlxnet/core/framework/tensor.h"
#include "dlxnet/core/framework/node_def.pb.h"
#include "dlxnet/core/framework/op_def.pb.h"
#include "dlxnet/core/framework/op.h"
#include "dlxnet/core/framework/node_def_util.h"
#include "dlxnet/core/platform/types.h"



namespace dlxnet{
    namespace shape_inference{
        // dimension, shape(list of dimension) and their handle(ptr)
        // all of them is just for simple.
        class InferenceContext;

        // Dimension values are accessed through InferenceContext.
        class Dimension {
            private:
                Dimension(int64 value);
                ~Dimension() {}
                const int64 value_;
                friend class InferenceContext;
                TF_DISALLOW_COPY_AND_ASSIGN(Dimension);

        };
        // their handle
        class DimensionHandle{
            public:
                DimensionHandle() {}
                bool SameHandle(DimensionHandle d) const { return ptr_ == d.ptr_; }
                std::size_t Handle() const { return reinterpret_cast<std::size_t>(ptr_); }
            private:
                DimensionHandle(const Dimension* dim) { ptr_ = dim; }
                const Dimension* operator->() const { return ptr_; }
                bool IsSet() const { return ptr_ != nullptr; }
                friend class InferenceContext;
                const Dimension* ptr_ = nullptr;
        };


        // list of dimension
        class Shape{
            private:
                Shape(const std::vector<DimensionHandle>& dims);
                Shape();
                ~Shape() {}

                const int32 rank_;
                const std::vector<DimensionHandle> dims_;
                friend class InferenceContext;
                TF_DISALLOW_COPY_AND_ASSIGN(Shape);
        };




        class ShapeHandle{
            public:
                ShapeHandle(){}
                bool SameHandle(ShapeHandle s)const{return ptr_==s.ptr_;}
                std::size_t Handle()const {return reinterpret_cast<std::size_t>(ptr_);}
            private:
                ShapeHandle(const Shape* shape) { ptr_ = shape; }
                const Shape* operator->() const { return ptr_; }
                bool IsSet() const { return ptr_ != nullptr; }
                friend class InferenceContext;
                const Shape* ptr_ = nullptr;
        };

        class InferenceContext{
            public:
                static constexpr int32 kUnknownRank = -1;
                InferenceContext(const NodeDef& node_def,
                        const OpDef& op_def);
                ~InferenceContext();
                // input();
                // output();
                Status Run(const std::function<Status(shape_inference::InferenceContext* c)>& fn);

                // accessor
                void SetInput(int idx, ShapeHandle shape){inputs_[idx] = shape;}
                ShapeHandle input(int64 idx) const { return inputs_[idx]; }
                int num_inputs() const { return inputs_.size(); }

                void set_output(int idx, ShapeHandle shape) { outputs_.at(idx) = shape; }
                int num_outputs() const { return outputs_.size(); }
                ShapeHandle output(int idx) const { return outputs_.at(idx); }
                // Describes the whole context, for debugging purposes.
                string DebugString() const;
            private:
                // Shared initialization across the two constructors.  Remove
                // once we get rid of one of them.
                void PreInputInit(const OpDef& op_def);
                std::vector<ShapeHandle> inputs_;
                std::vector<ShapeHandle> outputs_;
                NameRangeMap input_name_map_;
                NameRangeMap output_name_map_;

                // An error set during construction. TODO(cwhipkey): remove when test
                // constructor is removed.
                Status construction_status_;

                const NodeDef& node_def_;
                TF_DISALLOW_COPY_AND_ASSIGN(InferenceContext);
        };
        // -----------------------------------------------------------------------------
        // Template and inline method implementations, please ignore

        inline Shape::Shape(const std::vector<DimensionHandle>& dims)
            : rank_(dims.size()), dims_(dims) {}
        // unknown at present
        inline Shape::Shape()
            : rank_(InferenceContext::kUnknownRank){}

        inline Dimension::Dimension(int64 value) : value_(value) {
            DCHECK(value >= 0)
                << "Dimension must be non-negative or equal to "
                "InferenceContext::kUnknownDim but got "
                << value;
        }
    }




}


#endif

