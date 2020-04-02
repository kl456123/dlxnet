#ifndef DLXNET_CORE_FRAMEWORK_OP_KERNEL_H_
#define DLXNET_CORE_FRAMEWORK_OP_KERNEL_H_
#include <functional>
#include <memory>

#include <atomic>
#include <functional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dlxnet/core/framework/allocator.h"
#include "dlxnet/core/framework/device_base.h"
#include "dlxnet/core/framework/graph.pb.h"
#include "dlxnet/core/framework/kernel_def.pb.h"
#include "dlxnet/core/framework/kernel_def_builder.h"
#include "dlxnet/core/framework/node_def.pb.h"
#include "dlxnet/core/framework/node_def_util.h"
#include "dlxnet/core/framework/op.h"  // TODO(b/62899350): Remove
#include "dlxnet/core/framework/selective_registration.h"
#include "dlxnet/core/framework/session_state.h"
#include "dlxnet/core/framework/tensor.h"
#include "dlxnet/core/framework/tensor_shape.h"
#include "dlxnet/core/framework/tensor_shape.pb.h"  // TODO(b/62899350): Remove
#include "dlxnet/core/framework/tracking_allocator.h"
#include "dlxnet/core/framework/types.h"
#include "dlxnet/core/framework/types.pb.h"
#include "dlxnet/core/framework/function.h"
#include "dlxnet/core/lib/errors.h"
#include "dlxnet/core/lib/status.h"
#include "dlxnet/core/lib/gtl/array_slice.h"
#include "dlxnet/core/platform/env.h"
#include "dlxnet/core/platform/logging.h"
#include "dlxnet/core/platform/macros.h"
#include "dlxnet/core/platform/mutex.h"
#include "dlxnet/core/platform/thread_annotations.h"
#include "dlxnet/core/platform/types.h"
#include "dlxnet/core/protobuf/config.pb.h"


namespace dlxnet{
    class OpKernelConstruction;
    class OpKernelContext;
    class AsyncOpKernel;

    class OpKernel{
        public:
            // OpKernel won't be instantiated by the scheduler, so you may perform
            // expensive initialization in the descendant's constructor.
            explicit OpKernel(OpKernelConstruction* context);

            // Specialized constructor that enables the descendant to provide a different
            // `NodeDef` value. For example, this constructor can be used to provide a
            // stripped-down `NodeDef` that does not contain the full set of attrs (such
            // as tensor values) if the descendant stores them in a different form.
            explicit OpKernel(OpKernelConstruction* context,
                    std::unique_ptr<const NodeDef> node_def);


            virtual ~OpKernel();

            virtual void Compute(OpKernelContext* context)=0;

            // Returns nullptr iff this op kernel is synchronous.
            virtual AsyncOpKernel* AsAsync() { return nullptr; }
            virtual const AsyncOpKernel* AsAsync() const { return nullptr; }

            // Returns true iff this op kernel is considered "expensive". The
            // runtime may use this flag to optimize graph execution for example
            // to "inline" inexpensive kernels.
            virtual bool IsExpensive() {
                return expensive_;
            }

            // Accessors.
            const NodeDef& def() const { return *def_; }
            const string& name() const;              // Same as def().name()
            const string& type_string() const;       // Same as def().op()
            const string& requested_device() const;  // Same as def().device()

            int num_inputs() const { return input_types_.size(); }
            DataType input_type(int i) const { return input_types_[i]; }
            const DataTypeVector& input_types() const { return input_types_; }
            const MemoryTypeVector& input_memory_types() const {
                return input_memory_types_;
            }
            const string& requested_input(int i) const;  // Same as def().input(i)

            int num_outputs() const { return output_types_.size(); }
            DataType output_type(int o) const { return output_types_[o]; }
            const DataTypeVector& output_types() const { return output_types_; }
            const MemoryTypeVector& output_memory_types() const {
                return output_memory_types_;
            }

            Status InputRange(StringPiece input_name, int* start, int* stop) const;
            Status OutputRange(StringPiece output_name, int* start, int* stop) const;

        private:
            const std::unique_ptr<const NodeDef> def_;
            const DataTypeVector input_types_;
            const MemoryTypeVector input_memory_types_;
            const DataTypeVector output_types_;
            const MemoryTypeVector output_memory_types_;
            NameRangeMap input_name_map_;
            NameRangeMap output_name_map_;

            bool expensive_;

            TF_DISALLOW_COPY_AND_ASSIGN(OpKernel);

    };

    class AsyncOpKernel:public OpKernel{
        public:
            typedef std::function<void()> DoneCallback;
            virtual void ComputeAsync(OpKernelContext* context, DoneCallback done) = 0;
            void Compute(OpKernelContext* context) override;

            AsyncOpKernel* AsAsync() override { return this; }
            const AsyncOpKernel* AsAsync() const override { return this; }
    };


    class OpKernelConstruction{
        public:
            OpKernelConstruction(DeviceType device_type, DeviceBase* device,
                    Allocator* allocator, const NodeDef* node_def,
                    const OpDef* op_def, const DataTypeSlice& input_types,
                    const MemoryTypeSlice& input_memory_types, const DataTypeSlice& output_types,
                    const MemoryTypeSlice& output_memory_types, Status* status);

            Env* env() const { return device_->env(); }

            // Allocation of tensors during kernel construction:
            //
            // It is legal to temporarily allocate scratch tensor storage during
            // Op kernel construction. Scratch tensors should be allocated using
            // allocate_temp below. Some kernels need to keep tensors in between
            // invocations. If such a Tensor is allocated during kernel
            // construction this must be done using allocate_persistent, and the
            // Op may only store the returned PersistentTensor object. When the
            // Tensor is needed in a subsequent invocation, it can be retrieved
            // from the PersistentTensor using the AccessTensor method. This
            // ensures that the system is made aware of any use of the tensor's
            // allocated memory, which is needed for correctness on asynchronous
            // devices such as GPUs.

            // Allocates a temporary Tensor of the specified type and shape. The
            // Tensor must not be used after kernel construction is
            // complete. See comment above.
            Status allocate_temp(DataType type, const TensorShape& shape,
                    Tensor* out_temp);

            // User-supplied configuration of this operation.
            const NodeDef& def() const { return *def_; }

            // For inspecting the inputs to this operation.
            int num_inputs() const { return input_types_.size(); }
            DataType input_type(int i) const { return input_types_[i]; }
            const DataTypeSlice& input_types() const { return input_types_; }
            const MemoryTypeSlice& input_memory_types() const {
                return input_memory_types_;
            }

            // For inspecting the outputs expected from this operation.
            int num_outputs() const { return output_types_.size(); }
            DataType output_type(int i) const { return output_types_[i]; }
            const DataTypeSlice& output_types() const { return output_types_; }
            const MemoryTypeSlice& output_memory_types() const {
                return output_memory_types_;
            }

            // For recording configuration errors during construction.
            void SetStatus(const Status& status);
            const Status& status() const { return *status_; }

            // Look up the attr with name attr_name and set *value to its value.  If no
            // attr with attr_name is found in def(), or the attr does not have
            // a matching type, a non-ok status will be returned.
            template <class T>
                Status GetAttr(StringPiece attr_name, T* value) const;

            // Return true if the attr_name is defined in def().
            bool HasAttr(StringPiece attr_name) const;

            // Return the device type.
            const DeviceType& device_type() const { return device_type_; }

            // Helper routines for the OP_REQUIRES macros
            void CtxFailure(const Status& s);
            void CtxFailureWithWarning(const Status& s);
            void CtxFailure(const char* file, int line, const Status& s);
            void CtxFailureWithWarning(const char* file, int line, const Status& s);

            DeviceBase* device() const { return device_; }

        private:
            const DeviceType device_type_;
            DeviceBase* const device_;
            Allocator* allocator_;
            const NodeDef* def_;
            const OpDef* op_def_;
            DataTypeSlice input_types_;
            MemoryTypeSlice input_memory_types_;
            DataTypeSlice output_types_;
            MemoryTypeSlice output_memory_types_;
            Status* status_;

            // Allow op_def_ across from OpKernel, but not from subclasses.
            // TODO(irving): Remove protos from this header entirely.
            friend class OpKernel;

            TF_DISALLOW_COPY_AND_ASSIGN(OpKernelConstruction);
    };
    // TODO(mrry): Consider converting to a random_access_iterator, and upgrading
    // tensorflow::gtl::iterator_range to make the below container classes
    // unnecessary.
    template <typename ListType, typename ElementType>
        class OpArgIterator {
            public:
                using iterator_category = std::forward_iterator_tag;
                using value_type = ElementType;
                using pointer = ElementType*;
                using const_pointer = const ElementType*;
                using reference = ElementType&;
                using const_reference = const ElementType&;
                using difference_type = ptrdiff_t;

                OpArgIterator(const ListType* list, int i) : list_(list), i_(i) {}

                bool operator==(const OpArgIterator& rhs) {
                    DCHECK(list_ == rhs.list_);
                    return i_ == rhs.i_;
                }

                bool operator!=(const OpArgIterator& rhs) {
                    DCHECK(list_ == rhs.list_);
                    return i_ != rhs.i_;
                }

                OpArgIterator operator++() {  // prefix ++it
                    ++i_;
                    return *this;
                }

                OpArgIterator operator++(int) {  // postfix it++
                    OpArgIterator old_value = *this;
                    ++i_;
                    return old_value;
                }

                reference operator*() { return (*list_)[i_]; }
                pointer operator->() { return &(*list_)[i_]; }

                const_reference operator*() const { return (*list_)[i_]; }
                const_pointer operator->() const { return &(*list_)[i_]; }

            private:
                const ListType* const list_;
                int i_;
        };

    // Utility class for representing a list of immutable input tensors
    // that are passed to the op as a single named argument.
    class OpInputList {
        public:
            typedef OpArgIterator<OpInputList, const Tensor> Iterator;
            OpInputList() : ctx_(nullptr), start_(0), stop_(0) {}
            OpInputList(OpKernelContext* ctx, int start, int stop)
                : ctx_(ctx), start_(start), stop_(stop) {}
            OpInputList& operator=(const OpInputList& other) = default;
            const Tensor& operator[](int i) const;
            int size() const { return stop_ - start_; }
            Iterator begin() const { return Iterator(this, 0); }
            Iterator end() const { return Iterator(this, size()); }

        private:
            OpKernelContext* ctx_;  // not owned
            int start_;
            int stop_;
    };

    // Utility class for representing a list of mutable ("ref") input tensors
    // that are passed to the op as a single named argument.
    class OpMutableInputList {
        public:
            typedef OpArgIterator<OpMutableInputList, Tensor*> Iterator;
            OpMutableInputList(OpKernelContext* ctx, int start, int stop)
                : ctx_(ctx), start_(start), stop_(stop) {}
            OpMutableInputList() : ctx_(nullptr), start_(0), stop_(0) {}
            OpMutableInputList& operator=(const OpMutableInputList& other) = default;
            Tensor at(int i, bool lock_held);
            mutex* ref_mutex(int i);
            int size() const { return stop_ - start_; }
            Iterator begin() const { return Iterator(this, 0); }
            Iterator end() const { return Iterator(this, size()); }

        private:
            OpKernelContext* ctx_;  // not owned
            int start_;
            int stop_;
    };

    // Utility class for representing a list of output tensors that are
    // grouped as a single named output.
    class OpOutputList {
        public:
            typedef OpArgIterator<OpOutputList, const Tensor*> Iterator;
            OpOutputList() : ctx_(nullptr), start_(0), stop_(0) {}
            OpOutputList(OpKernelContext* ctx, int start, int stop)
                : ctx_(ctx), start_(start), stop_(stop) {}
            OpOutputList& operator=(const OpOutputList& other) = default;
            Tensor* operator[](int i);
            bool required(int i) const;
            DataType expected_output_dtype(int i) const;
            Status allocate(int i, const TensorShape& shape, Tensor** output);
            void set(int i, const Tensor& tensor);
            void set_ref(int i, mutex* mu, Tensor* tensor_for_ref);
            int size() const { return stop_ - start_; }
            Iterator begin() const { return Iterator(this, 0); }
            Iterator end() const { return Iterator(this, size()); }

        private:
            OpKernelContext* ctx_;  // not owned
            int start_;
            int stop_;
    };

    // Holds a tensor or tensor reference. For tensor references, we need
    // a mutex to prevent concurrent access to the tensor.
    struct TensorValue {
        TensorValue() : mutex_if_ref(nullptr), tensor(nullptr) {}
        explicit TensorValue(Tensor* t) : mutex_if_ref(nullptr), tensor(t) {}
        TensorValue(mutex* mu, Tensor* t) : mutex_if_ref(mu), tensor(t) {}
        Tensor* operator->() const { return tensor; }
        bool is_ref() const { return mutex_if_ref != nullptr; }

        // Return the dtype of the Tensor. For references, return the underlying type.
        DataType dtype() const {
            if (is_ref()) {
                return MakeRefType(tensor->dtype());
            } else {
                return tensor->dtype();
            }
        }

        // Return the dtype of the Tensor. For references, return the underlying type.
        // This variation on the dtype() acquires the lock for references.
        //
        // TODO(b/133843385): Disallow dtype modifications
        DataType dtype_safe() const {
            if (is_ref()) {
                tf_shared_lock ml(*mutex_if_ref);
                return MakeRefType(tensor->dtype());
            } else {
                return tensor->dtype();
            }
        }

        mutex* mutex_if_ref;  // nullptr if not a ref, != nullptr if a ref
        Tensor* tensor;
    };


    class OpKernelContext{
        public:
            // The first element of a WrappedAllocator is a "base" Allocator and
            // the second element is that Allocator wrapped by a
            // TrackingAllocator
            typedef std::pair<Allocator*, TrackingAllocator*> WrappedAllocator;

            struct Params {
                // The step being executed.
                int64 step_id = 0;

                // The op kernel being computed.
                OpKernel* op_kernel = nullptr;

                // The device on which the kernel is running.
                DeviceBase* device = nullptr;

                bool track_allocations = false;
                bool log_memory = false;

                // Array indexed by output number for this node
                const AllocatorAttributes* output_attr_array = nullptr;

                // The session state for this op.
                SessionState* session_state = nullptr;

                // Unique session identifier. Can be empty.
                string session_handle;

                // Inputs to this op kernel.
                const gtl::InlinedVector<TensorValue, 4>* inputs = nullptr;
                bool is_input_dead = false;

                const gtl::InlinedVector<AllocatorAttributes, 4>* input_alloc_attrs =
                    nullptr;

                // Device context.
                DeviceContext* op_device_context = nullptr;

                // Function call supports.
                CallFrameInterface* call_frame = nullptr;

                std::function<void(std::function<void()>)>* runner = nullptr;
            };

            // params must outlive the OpKernelContext.
            explicit OpKernelContext(Params* params);
            OpKernelContext(Params* params, int num_outputs);
            ~OpKernelContext();

            Env* env() const { return params_->device->env(); }

            int64 step_id() const { return params_->step_id; }

            const OpKernel& op_kernel() const { return *params_->op_kernel; }

            // Input/output signature.

            int num_inputs() const { return params_->inputs->size(); }
            DataType input_dtype(int index) const;
            Status input_dtype(StringPiece name, DataType* dtype) const;
            MemoryType input_memory_type(int index) const;

            int num_outputs() const { return outputs_.size(); }
            DataType expected_output_dtype(int index) const;
            MemoryType output_memory_type(int index) const;

            // Returns an immutable input tensor. May only be used for non-Ref
            // inputs. For Ref inputs use mutable_input below.
            // REQUIRES: !IsRefType(input_dtype(index))
            // TODO(mrry): Convert this to return Status.
            const Tensor& input(int index);

            // Returns the named immutable input tensor in "tensor", as defined
            // in the OpDef. May only be used for non-Ref inputs. For Ref inputs
            // use mutable_input below.
            // REQUIRES: !IsRefType(input_dtype(index))
            // REQUIRES: the named input must not be a list.
            Status input(StringPiece name, const Tensor** tensor);

            // Returns the named list-valued immutable input in "list", as
            // defined in the OpDef.  If the named output is not list-valued,
            // returns a one-element list. May only be used for non-Ref
            // inputs. For Ref inputs use mutable_input below.
            // REQUIRES: !IsRefType(input_dtype(index))
            Status input_list(StringPiece name, OpInputList* list);

            // For mutable inputs, use the following together to make sure there
            // is no concurrent access to mutable_input(), e.g.:
            // {
            //   Tensor& t = context->mutable_input(index);
            //   mutex_lock lock(*context->input_ref_mutex(index));
            //   // modify the values in t
            // }
            // REQUIRES: IsRefType(input_dtype(index))
            Status input_ref_mutex(StringPiece name, mutex** out_mutex);

            // Returns a mutable input tensor. Must be used to access Ref
            // inputs.  REQUIRES: IsRefType(input_dtype(index)). The caller may
            // modify the values stored in the Tensor buffer, and modifications
            // will be visible to other Ops reading the same ref tensor. If
            // !lock_held the input mutex will be acquired before returning the
            // Tensor.
            // TODO(mrry): Convert this to return Status.
            Tensor mutable_input(int index, bool lock_held);

            // Returns the named mutable input tensor in "tensor", as defined in
            // the OpDef. Must be used to access Ref inputs. The values stored
            // in the Tensor buffer may be modified, and modifications will be
            // visible to other Ops reading the same ref tensor. If !lock_held
            // the input mutex will be acquired before returning the Tensor.
            // REQUIRES: the named input must not be a list.
            // REQUIRES: the named input must be a ref tensor.
            Status mutable_input(StringPiece name, Tensor* tensor, bool lock_held);

            // Returns the named list-valued mutable input in "list", as defined
            // in the OpDef.  If the named input is not list-valued, returns a
            // one-element list. Must be used to access Ref inputs. The values
            // stored in the Tensor buffer may be modified, and modifications
            // will be visible to other Ops reading the same ref tensor.
            // REQUIRES: the named input must be a ref tensor.
            Status mutable_input_list(StringPiece name, OpMutableInputList* list);

            // Allocation of tensors during kernel execution inside the Compute
            // method:
            //
            // There are three methods to allocate Tensors when an Op kernel
            // executes.
            //
            // 1) allocate_persistent. This is only needed for Tensors that will
            // be stored by the Op between invocations, and it *must* be used
            // for those Tensors. The call returns a PersistentTensor, and that
            // is the only object the Op is allowed to hold on to between
            // invocations. When the Tensor is needed in a subsequent
            // invocation, it can be retrieved from the PersistentTensor using
            // the AccessTensor method. This ensures that the system is made
            // aware of any use of the tensor's allocated memory, which is
            // needed for correctness on asynchronous devices such as GPUs.
            //
            // 2) allocate_output. This should be used to allocate any tensor
            // that is going to be used as an output from the Op at the end of
            // the current execution. The caller indicates which output the
            // Tensor will be assigned to, and the call returns the
            // newly-allocated Tensor. The Tensor can subsequently be assigned
            // to during kernel execution, and will be used as the designated
            // output when the kernel execution completes.
            //
            // 3) allocate_temp. This should be used to allocate any scratch
            // storage that is needed while the kernel is executing, and will
            // not be retained by the Op.
            //
            // In some cases a Tensor needs to be used as an output even though
            // it was previously allocated elsewhere. The Tensor may have been
            // passed as an input, or stored in a PersistentTensor during a
            // previous kernel execution, or allocated earlier in the kernel
            // execution at a time when it was not known which output it would
            // be assigned to. In this case the kernel can use set_output or
            // set_output_ref to indicate that the tensor should be used as the
            // designated output. It is legal to use any previously-allocated
            // Tensor as an argument to set_output or set_output_ref, including
            // Tensors allocated via allocate_temp. There may be a performance
            // penalty to using a Tensor that was not allocated using
            // allocate_output. This is because allocate_output uses the
            // AllocatorAttributes stored in output_attr_array for the
            // designated output. In some cases, using the wrong attributes may
            // cause an extra copy of the Tensor's buffer.

            // Allocates output for the specified output index with shape.
            // OpKernelContext retains ownership of the returned pointer. See
            // comment above.
            //
            // If memory allocation fails, returns an error status.
            //
            // REQUIRES: !IsRefType(expected_output_dtype(index))
            Status allocate_output(int index, const TensorShape& shape,
                    Tensor** tensor) TF_MUST_USE_RESULT;
            Status allocate_output(StringPiece name, const TensorShape& shape,
                    Tensor** tensor) TF_MUST_USE_RESULT;
            // The following methods use the supplied attributes instead of
            // those in output_attr_array. The caller is responsible for
            // ensuring that the attributes are "compatible" with the
            // output_attr_array, e.g. the tensor is allocated on the correct
            // device. See comment above.
            Status allocate_output(int index, const TensorShape& shape, Tensor** tensor,
                    AllocatorAttributes attr) TF_MUST_USE_RESULT;
            Status allocate_output(StringPiece name, const TensorShape& shape,
                    Tensor** tensor,
                    AllocatorAttributes attr) TF_MUST_USE_RESULT;

            // Allocates a temporary Tensor of the specified type and
            // shape. Devices such as GPUs that enqueue Ops for lazy execution
            // may retain references to the temporary tensors after the Op's
            // Compute method has run. See comment above.
            Status allocate_temp(DataType type, const TensorShape& shape,
                    Tensor* out_temp, AllocatorAttributes allocator_attr,
                    const AllocationAttributes& allocation_attr);
            Status allocate_temp(DataType type, const TensorShape& shape,
                    Tensor* out_temp, AllocatorAttributes allocator_attr) {
                return allocate_temp(type, shape, out_temp, allocator_attr,
                        AllocationAttributes());
            }
            Status allocate_temp(DataType type, const TensorShape& shape,
                    Tensor* out_temp) {
                return allocate_temp(type, shape, out_temp, AllocatorAttributes());
            }


            // Copies a tensor (allocated by the caller) to the specified output
            // index.  REQUIRES: !IsRefType(expected_output_dtype(index))
            // REQUIRES: 'tensor' must have the same MemoryType as
            // output_memory_types[index]. See comment above.
            Status set_output(StringPiece name, const Tensor& tensor);

            // To output a reference.  Caller retains ownership of mu and tensor_for_ref,
            // and they must outlive all uses within the step. See comment above.
            // REQUIRES: IsRefType(expected_output_dtype(index))
            Status set_output_ref(StringPiece name, mutex* mu, Tensor* tensor_for_ref);

            // Returns nullptr if allocate_output() or set_output() have not been called.
            Status mutable_output(StringPiece name, Tensor** tensor);

            // Return the DeviceContext that should be used for this Op.
            //
            // If using the templated function, the type must be a subclass
            // of DeviceContext.
            //
            // Returns nullptr if the device did not provide one.
            template <typename T>
                T* op_device_context();
            DeviceContext* op_device_context() {
                DeviceContext* ret = params_->op_device_context;
                // if (ret == nullptr) {
                // auto* dev_info = device()->tensorflow_gpu_device_info();
                // if (dev_info) ret = dev_info->default_context;
                // }
                return ret;
            }

            AllocatorAttributes input_alloc_attr(int index) const {
                if (params_->input_alloc_attrs == nullptr) {
                    return AllocatorAttributes();
                } else {
                    DCHECK_GE(index, 0);
                    DCHECK_LT(index, params_->input_alloc_attrs->size());
                    return (*params_->input_alloc_attrs)[index];
                }
            }

            AllocatorAttributes output_alloc_attr(int index) const {
                return params_->output_attr_array[index];
            }

            // An op kernel can access the session state it belongs to.
            SessionState* session_state() const { return params_->session_state; }

            // Unique identifier of the session it belongs to. Can be empty.
            string session_handle() const { return params_->session_handle; }

            // Function call support.
            //
            // If this kernel invocation is within a function execution,
            // call_frame() returns the call frame for the function call.
            CallFrameInterface* call_frame() const { return params_->call_frame; }

            std::function<void(std::function<void()>)>* runner() const {
                return params_->runner;
            }

            // eigen utils
            // OpKernels can use these eigen devices to carry out their
            // numerical computation.
            const Eigen::ThreadPoolDevice& eigen_cpu_device() const {
                return *device()->eigen_cpu_device();
            }

            template <typename EigenDeviceType>
                const EigenDeviceType& eigen_device() const;

            // An OpKernel should call SetStatus() if Compute() encounters an
            // error.
            void SetStatus(const Status& status);
            const Status& status() const { return status_; }

            // Other accessors.

            bool is_input_dead() const { return params_->is_input_dead; }

            // May be used, e.g., to get GPU handles, etc.
            // TODO(tucker): Add example usage.
            DeviceBase* device() const { return params_->device; }

            // Helper routines for the OP_REQUIRES macros
            void CtxFailure(const Status& s);
            void CtxFailureWithWarning(const Status& s);
            void CtxFailure(const char* file, int line, const Status& s);
            void CtxFailureWithWarning(const char* file, int line, const Status& s);

            // Unrecommended functions: these are functions that have some
            // current uses but are not recommended for use, and may go away at
            // some future major version release.
            //
            // The following functions all have versions that return Status
            // to capture error conditions, and are strongly preferred.
            Tensor* mutable_output(int index);
            void set_output(int index, const Tensor& tensor);
            mutex* input_ref_mutex(int index);
            void set_output_ref(int index, mutex* mu, Tensor* tensor_for_ref);
            TensorValue release_output(int index);

            bool track_allocations() const { return params_->track_allocations; }

            // Records temp memory allocation. Tensor object is recorded to identify the
            // case where temp memory is used as output memory.
            void record_temp_memory_allocation(int64 size, const Tensor& t)
                LOCKS_EXCLUDED(tracking_state_->stats_mu);

            // Returns recorded size of temporary memory;
            int64 temp_memory_allocated() const LOCKS_EXCLUDED(tracking_state_->stats_mu);

            // Records persistent memory allocation, size can be negative indicating
            // deallocation.
            void record_persistent_memory_allocation(int64 size, int64 alloc_id = -1)
                LOCKS_EXCLUDED(tracking_state_->stats_mu);

            // Returns recorded size and ids of persistent memory.
            int64 persistent_memory_allocated() const
                LOCKS_EXCLUDED(tracking_state_->stats_mu);

            std::vector<int64> persistent_alloc_ids() const
                LOCKS_EXCLUDED(tracking_state_->stats_mu);

            // Resets counters for temp and persistent memory and recorded ids.
            void clear_recorded_memory() LOCKS_EXCLUDED(tracking_state_->stats_mu);

            bool input_is_ref(int index) const;

            void set_record_memory_consumption(bool v);
            Allocator* get_allocator(AllocatorAttributes attr);
        private:
            bool record_memory_consumption_ = false;
            void record_tensor_reference(const Tensor& tensor);

            // Internal common method used when allocating tensor memory
            Status allocate_tensor(DataType type, const TensorShape& shape,
                    Tensor* out_tensor,
                    AllocatorAttributes allocator_attr) {
                return allocate_tensor(type, shape, out_tensor, allocator_attr,
                        AllocationAttributes());
            }

            Status allocate_tensor(DataType type, const TensorShape& shape,
                    Tensor* out_tensor, AllocatorAttributes allocator_attr,
                    const AllocationAttributes& allocation_attr);
            Status status_;
            Params* params_;                  // not owned
            gtl::InlinedVector<TensorValue, 4> outputs_;

            // The following data members are only used when allocation tracking is
            // enabled, memory consumption is being recorded, or tensor access is being
            // recorded.
            struct TrackingState {
                mutable mutex mu;
                gtl::InlinedVector<WrappedAllocator, 4> wrapped_allocators GUARDED_BY(mu);

                mutable mutex stats_mu;
                int64 temp_memory_allocated GUARDED_BY(stats_mu) = 0;

                int64 persistent_memory_allocated GUARDED_BY(stats_mu) = 0;
                gtl::InlinedVector<std::pair<const void*, int64>, 2>
                    temp_tensor_buffer_and_size GUARDED_BY(stats_mu);
                gtl::InlinedVector<int64, 2> persistent_alloc_ids GUARDED_BY(stats_mu);
            };
            std::unique_ptr<TrackingState> tracking_state_;

            TF_DISALLOW_COPY_AND_ASSIGN(OpKernelContext);
    };

    template <>
        const Eigen::ThreadPoolDevice& OpKernelContext::eigen_device() const;

    // Register your OpKernel by specifying the Op's name, the device the
    // kernel runs on, any type attr constraints for this kernel, any
    // host-memory args, and the class to instantiate.  Examples:
    //
    //  // A kernel that supports all types.
    //  REGISTER_KERNEL_BUILDER(Name("Save").Device(DEVICE_CPU), SaveOp);
    //
    //  // The following are equivalent ways of specifying that the kernel only
    //  // works if the "T" type attr is set to DT_FLOAT.
    //  REGISTER_KERNEL_BUILDER(
    //      Name("Sub").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    //      SubOp<float>);
    //  // (You would then repeat this for every type supported by "Sub".)
    //
    //  // This form allows you to specify a list of types as the constraint.
    //  REGISTER_KERNEL_BUILDER(Name("Sub")
    //                              .Device(DEVICE_CPU)
    //                              .TypeConstraint("T", {DT_FLOAT}),
    //                          SubOp<float>);
    //
    //  // A kernel that expects one of the input tensors in host memory.
    //  REGISTER_KERNEL_BUILDER(
    //      Name("Reshape").Device(DEVICE_GPU).HostMemory("shape"), ReshapeOp);
    //
    // See kernel_def_builder for details.

    // Instantiate an OpKernel that has been registered.  Returns nullptr
    // if no operation for that type of device / input signature combination
    // (and a NOT_FOUND *status), or there is an error in construction (and
    // an INVALID_ARGUMENT *status).  Otherwise, the caller takes ownership
    // of the returned pointer.
    // EXPECTED USAGE: unique_ptr<OpKernel> op = CreateOpKernel(...);
    // REQUIRES: def has all attrs specified (e.g. using AddDefaultsToNodeDef()).
    std::unique_ptr<OpKernel> CreateOpKernel(DeviceType device_type,
            DeviceBase* device,
            Allocator* allocator,
            const NodeDef& def,
            Status* status);
    Status CreateOpKernel(DeviceType device_type, DeviceBase* device,
            Allocator* allocator,
            const NodeDef& def,
            OpKernel** kernel);

    // Returns a message with a description of the kernels registered for op
    // `op_name`.
    string KernelsRegisteredForOp(StringPiece op_name);
    // Gets a list of all registered kernels for a given op
    KernelList GetRegisteredKernelsForOp(StringPiece op_name);

    // Gets a list of all registered kernels.
    KernelList GetAllRegisteredKernels();

    // Gets a list of all registered kernels for which predicate returns true
    KernelList GetFilteredRegisteredKernels(
            const std::function<bool(const KernelDef&)>& predicate);

    // Writes a list of all registered kernels to LOG(INFO), to help users debug
    // missing kernel errors.
    void LogAllRegisteredKernels();

    // -----------------------------------------------------------------------------
    // OpKernel registration implementation follows, please ignore.

    // Allow the REGISTER_KERNEL_BUILDER(Name("op_name").Device(...)...) syntax.
    //
    namespace register_kernel {
        class Name : public KernelDefBuilder {
            public:
                // With selective registration, kernels whose implementation class is not used
                // by any kernel are disabled with the SHOULD_REGISTER_OP_KERNEL call in
                // REGISTER_KERNEL_BUILDER_UNIQ. However, an unused kernel that shares an
                // implementation class with a used kernel would get through that mechanism.
                //
                // This mechanism stops that registration by changing the name of the kernel
                // for the unused op to one that is ignored by
                // OpKernelRegistrar::InitInternal.  Note that this method alone is
                // not sufficient - the compiler can't evaluate the entire KernelDefBuilder at
                // compilation time, so this method doesn't actually reduce code size.
                explicit Name(const char* op)
                    : KernelDefBuilder(SHOULD_REGISTER_OP(op) ? op : "_no_register") {}
        };


        namespace system {

            class Name : public KernelDefBuilder {
                public:
                    // For system kernels, we ignore selective registration and
                    // unconditionally register the kernel.
                    explicit Name(const char* op) : KernelDefBuilder(op) {}
            };

        }  // namespace system

    }  // namespace register_kernel

#define REGISTER_KERNEL_BUILDER(kernel_builder, ...) \
    REGISTER_KERNEL_BUILDER_UNIQ_HELPER(__COUNTER__, kernel_builder, __VA_ARGS__)

#define REGISTER_KERNEL_BUILDER_UNIQ_HELPER(ctr, kernel_builder, ...) \
    REGISTER_KERNEL_BUILDER_UNIQ(ctr, kernel_builder, __VA_ARGS__)

#define REGISTER_KERNEL_BUILDER_UNIQ(ctr, kernel_builder, ...)        \
    constexpr bool should_register_##ctr##__flag =                      \
    SHOULD_REGISTER_OP_KERNEL(#__VA_ARGS__);                        \
    static ::dlxnet::kernel_factory::OpKernelRegistrar              \
    registrar__body__##ctr##__object(                               \
            should_register_##ctr##__flag                               \
            ? ::dlxnet::register_kernel::kernel_builder.Build() \
            : nullptr,                                              \
#__VA_ARGS__,                                               \
            [](::dlxnet::OpKernelConstruction* context)             \
            -> ::dlxnet::OpKernel* {                            \
            return new __VA_ARGS__(context);                          \
            });

    // The `REGISTER_SYSTEM_KERNEL_BUILDER()` macro acts as
    // `REGISTER_KERNEL_BUILDER()` except that the kernel is registered
    // unconditionally even when selective registration is used.
#define REGISTER_SYSTEM_KERNEL_BUILDER(kernel_builder, ...)               \
    REGISTER_SYSTEM_KERNEL_BUILDER_UNIQ_HELPER(__COUNTER__, kernel_builder, \
            __VA_ARGS__)

#define REGISTER_SYSTEM_KERNEL_BUILDER_UNIQ_HELPER(ctr, kernel_builder, ...) \
    REGISTER_SYSTEM_KERNEL_BUILDER_UNIQ(ctr, kernel_builder, __VA_ARGS__)

#define REGISTER_SYSTEM_KERNEL_BUILDER_UNIQ(ctr, kernel_builder, ...)    \
    static ::dlxnet::kernel_factory::OpKernelRegistrar                 \
    registrar__body__##ctr##__object(                                  \
            ::dlxnet::register_kernel::system::kernel_builder.Build(), \
#__VA_ARGS__,                                                  \
            [](::dlxnet::OpKernelConstruction* context)                \
            -> ::dlxnet::OpKernel* {                               \
            return new __VA_ARGS__(context);                             \
            });

    namespace kernel_factory {

        // OpKernelFactory is responsible for creating OpKernels when TensorFlow needs
        // them. You register factories with the TensorFlow core by constructing an
        // OpKernelRegistrar and passing the factory as a constructor parameter.
        class OpKernelFactory {
            public:
                virtual OpKernel* Create(OpKernelConstruction* context) = 0;
                virtual ~OpKernelFactory() = default;
        };

        class OpKernelRegistrar {
            public:
                // Registers the given kernel factory with TensorFlow. TF will call the
                // factory Create() method when it determines that a kernel matching the given
                // KernelDef is required.
                OpKernelRegistrar(const KernelDef* kernel_def, StringPiece kernel_class_name,
                        std::unique_ptr<OpKernelFactory> factory) {
                    // Perform the check in the header to allow compile-time optimization
                    // to a no-op, allowing the linker to remove the kernel symbols.
                    if (kernel_def != nullptr) {
                        InitInternal(kernel_def, kernel_class_name, std::move(factory));
                    }
                }

                // Registers the given factory function with TensorFlow. This is equivalent
                // to registering a factory whose Create function invokes `create_fn`.
                OpKernelRegistrar(const KernelDef* kernel_def, StringPiece kernel_class_name,
                        OpKernel* (*create_fn)(OpKernelConstruction*)) {
                    // Perform the check in the header to allow compile-time optimization
                    // to a no-op, allowing the linker to remove the kernel symbols.
                    if (kernel_def != nullptr) {
                        InitInternal(kernel_def, kernel_class_name,
                                absl::make_unique<PtrOpKernelFactory>(create_fn));
                    }
                }

            private:
                struct PtrOpKernelFactory : public OpKernelFactory {
                    explicit PtrOpKernelFactory(OpKernel* (*create_func)(OpKernelConstruction*))
                        : create_func_(create_func) {}

                    OpKernel* Create(OpKernelConstruction* context) override;

                    OpKernel* (*create_func_)(OpKernelConstruction*);
                };

                void InitInternal(const KernelDef* kernel_def, StringPiece kernel_class_name,
                        std::unique_ptr<OpKernelFactory> factory);
        };

    }  // namespace kernel_factory

            // -----------------------------------------------------------------------------
            // Template and inline method implementations, please ignore

            template <class T>
                Status OpKernelConstruction::GetAttr(StringPiece attr_name, T* value) const {
                    return GetNodeAttr(def(), attr_name, value);
                }

            inline DataType OpKernelContext::input_dtype(int index) const {
                DCHECK_GE(index, 0);
                DCHECK_LT(index, num_inputs());
                const TensorValue& value((*params_->inputs)[index]);
                return value.dtype();
            }

            inline Tensor* OpKernelContext::mutable_output(int index) {
                DCHECK_GE(index, 0);
                DCHECK_LT(index, num_outputs());
                // No need to record_tensor_reference since the output must already
                // have been set by a call that did so.
                return outputs_[index].tensor;
            }

            inline TensorValue OpKernelContext::release_output(int index) {
                DCHECK_GE(index, 0);
                DCHECK_LT(index, num_outputs());
                TensorValue value = outputs_[index];
                outputs_[index] = TensorValue();
                return value;
            }

            inline MemoryType OpKernelContext::input_memory_type(int index) const {
                DCHECK_GE(index, 0);
                DCHECK_LT(index, num_inputs());
                return op_kernel().input_memory_types()[index];
            }

            inline DataType OpKernelContext::expected_output_dtype(int index) const {
                DCHECK_GE(index, 0);
                DCHECK_LT(index, num_outputs());
                return params_->op_kernel->output_type(index);
            }

            inline MemoryType OpKernelContext::output_memory_type(int index) const {
                DCHECK_GE(index, 0);
                DCHECK_LT(index, num_outputs());
                return op_kernel().output_memory_types()[index];
            }

            inline void OpKernelContext::record_tensor_reference(const Tensor& tensor) {
                // DCHECK_EQ(params_->device->RequiresRecordingAccessedTensors(),
                // params_->record_tensor_accesses);
                // if (params_->record_tensor_accesses) {
                // really_record_tensor_reference(tensor);
                // }
            }

            inline bool OpKernelContext::input_is_ref(int index) const {
                const TensorValue& value((*params_->inputs)[index]);
                return value.is_ref();
            }


            inline mutex* OpKernelContext::input_ref_mutex(int index) {
                DCHECK_GE(index, 0);
                DCHECK_LT(index, num_inputs());
                DCHECK(input_is_ref(index));
                return (*params_->inputs)[index].mutex_if_ref;
            }

            template <typename T>
                T* OpKernelContext::op_device_context() {
                    static_assert(std::is_base_of<DeviceContext, T>::value,
                            "T is not a subclass of DeviceContext");
                    return static_cast<T*>(op_device_context());
                }

            inline const Tensor& OpInputList::operator[](int i) const {
                DCHECK_GE(i, 0);
                DCHECK_LT(i, stop_ - start_);
                return ctx_->input(start_ + i);
            }

            inline mutex* OpMutableInputList::ref_mutex(int i) {
                DCHECK_GE(i, 0);
                DCHECK_LT(i, stop_ - start_);
                return ctx_->input_ref_mutex(start_ + i);
            }

            inline Tensor OpMutableInputList::at(int i, bool lock_held) {
                DCHECK_GE(i, 0);
                DCHECK_LT(i, stop_ - start_);
                return ctx_->mutable_input(start_ + i, lock_held);
            }

            inline Tensor* OpOutputList::operator[](int i) {
                DCHECK_GE(i, 0);
                DCHECK_LT(i, stop_ - start_);
                return ctx_->mutable_output(start_ + i);
            }


            inline DataType OpOutputList::expected_output_dtype(int i) const {
                DCHECK_GE(i, 0);
                DCHECK_LT(i, stop_ - start_);
                return ctx_->expected_output_dtype(start_ + i);
            }

            inline Status OpOutputList::allocate(int i, const TensorShape& shape,
                    Tensor** output) {
                DCHECK_GE(i, 0);
                DCHECK_LT(i, stop_ - start_);
                return ctx_->allocate_output(start_ + i, shape, output);
            }

            inline void OpOutputList::set(int i, const Tensor& tensor) {
                DCHECK_GE(i, 0);
                DCHECK_LT(i, stop_ - start_);
                ctx_->set_output(start_ + i, tensor);
            }

            inline void OpOutputList::set_ref(int i, mutex* mu, Tensor* tensor_for_ref) {
                DCHECK_GE(i, 0);
                DCHECK_LT(i, stop_ - start_);
                ctx_->set_output_ref(i, mu, tensor_for_ref);
            }

            // Convenience macros for asserting and handling exceptional conditions.
            // Analogous to the CHECK* macros provided by logging.h.
            //
            // Example use:
            // void Compute(OperationContext* context) {
            //   OP_REQUIRES(context, context->num_inputs() == 2,
            //               errors::InvalidArgument("FooOp requires 2 arguments"));
            //   ...
            //   Status status = SomeUncertainMethod();
            //   OP_REQUIRES_OK(context, status);
            //   ...
            // }

            // Generate a fatal error if OP_REQUIRES or OP_REQUIRES_OK are used in
            // AsyncOpKernel implementations. If these macros are used and the condition
            // does not hold, the `done` callback will never be called and the system will
            // deadlock, so a crash failure is preferable. Since the OP_REQUIRES[_OK] macros
            // are legal to use in AsyncOpKernel constructors, we use overload resolution
            // to distinguish between OpKernelConstruction* and OpKernelContext* context
            // types.
            inline void CheckNotInComputeAsync(OpKernelConstruction*, const char*) {}
            void CheckNotInComputeAsync(OpKernelContext* ctx,
                    const char* correct_macro_name);
#define OP_REQUIRES(CTX, EXP, STATUS)                     \
            do {                                                    \
                if (!TF_PREDICT_TRUE(EXP)) {                          \
                    CheckNotInComputeAsync((CTX), "OP_REQUIRES_ASYNC"); \
                    (CTX)->CtxFailure(__FILE__, __LINE__, (STATUS));    \
                    return;                                             \
                }                                                     \
            } while (0)

#define OP_REQUIRES_OK(CTX, ...)                             \
            do {                                                       \
                ::dlxnet::Status _s(__VA_ARGS__);                    \
                if (!TF_PREDICT_TRUE(_s.ok())) {                         \
                    CheckNotInComputeAsync((CTX), "OP_REQUIRES_OK_ASYNC"); \
                    (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s);  \
                    return;                                                \
                }                                                        \
            } while (0)

#define OP_REQUIRES_ASYNC(CTX, EXP, STATUS, CALLBACK)  \
            do {                                                 \
                if (!TF_PREDICT_TRUE(EXP)) {                       \
                    (CTX)->CtxFailure(__FILE__, __LINE__, (STATUS)); \
                    (CALLBACK)();                                    \
                    return;                                          \
                }                                                  \
            } while (0)

#define OP_REQUIRES_OK_ASYNC(CTX, STATUS, CALLBACK)         \
            do {                                                      \
                ::dlxnet::Status _s(STATUS);                        \
                if (!TF_PREDICT_TRUE(_s.ok())) {                        \
                    (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s); \
                    (CALLBACK)();                                         \
                    return;                                               \
                }                                                       \
            } while (0)


}// namespace dlxnet


#endif
