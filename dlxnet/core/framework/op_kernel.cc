#include "dlxnet/core/framework/op_kernel.h"

#include <cstdlib>
#include <cstring>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dlxnet/core/framework/allocation_description.pb.h"
#include "dlxnet/core/framework/attr_value_util.h"
#include "dlxnet/core/framework/device_attributes.pb.h"
#include "dlxnet/core/framework/graph.pb.h"
#include "dlxnet/core/framework/kernel_def.pb.h"
#include "dlxnet/core/framework/kernel_def_util.h"
#include "dlxnet/core/framework/log_memory.h"
// #include "dlxnet/core/framework/memory_types.h"
#include "dlxnet/core/framework/node_def.pb.h"
#include "dlxnet/core/framework/node_def_util.h"
#include "dlxnet/core/framework/op_def_util.h"
#include "dlxnet/core/framework/types.h"
#include "dlxnet/core/graph/graph.h"
#include "dlxnet/core/lib/core/errors.h"
#include "dlxnet/core/lib/core/notification.h"
#include "dlxnet/core/lib/core/stringpiece.h"
#include "dlxnet/core/lib/gtl/map_util.h"
#include "dlxnet/core/lib/io/path.h"
#include "dlxnet/core/lib/strings/str_util.h"
#include "dlxnet/core/lib/strings/strcat.h"
#include "dlxnet/core/platform/cpu_info.h"
#include "dlxnet/core/platform/env.h"
#include "dlxnet/core/platform/logging.h"
#include "dlxnet/core/platform/mutex.h"
#include "dlxnet/core/platform/types.h"
#include "dlxnet/core/util/ptr_util.h"

namespace dlxnet{
    // OpKernel ------------------------------------------------------------------

    OpKernel::OpKernel(OpKernelConstruction* context)
        : OpKernel(context, MakeUnique<const NodeDef>(context->def())) {}

    OpKernel::OpKernel(OpKernelConstruction* context,
            std::unique_ptr<const NodeDef> node_def)
        : def_(std::move(node_def)),
        input_types_(context->input_types().begin(),
                context->input_types().end()),
        input_memory_types_(context->input_memory_types().begin(),
                context->input_memory_types().end()),
        output_types_(context->output_types().begin(),
                context->output_types().end()),
        output_memory_types_(context->output_memory_types().begin(),
                context->output_memory_types().end()),
        input_name_map_(context->num_inputs()),
        output_name_map_(context->num_outputs()){
            OP_REQUIRES_OK(context,
                    NameRangesForNode(*def_, *context->op_def_, &input_name_map_,
                        &output_name_map_));

            // Kernels executing on GPU/SYCL tie very few resources on the CPU where the
            // scheduler runs: we consider them as inexpensive.
            expensive_ = context->device_type() != DeviceType(DEVICE_GPU) &&
                context->device_type() != DeviceType(DEVICE_SYCL);
        }

    OpKernel::~OpKernel() {}

    const string& OpKernel::name() const { return def_->name(); }
    const string& OpKernel::type_string() const { return def_->op(); }
    const string& OpKernel::requested_device() const { return def_->device(); }
    const string& OpKernel::requested_input(int i) const { return def_->input(i); }

    Status OpKernel::InputRange(StringPiece input_name, int* start,
            int* stop) const {
        const auto result = input_name_map_.find(input_name);
        if (result == input_name_map_.end()) {
            return errors::InvalidArgument("Unknown input name: ", input_name);
        } else {
            *start = result->second.first;
            *stop = result->second.second;
            return Status::OK();
        }
    }

    Status OpKernel::OutputRange(StringPiece output_name, int* start,
            int* stop) const {
        const auto result = output_name_map_.find(output_name);
        if (result == output_name_map_.end()) {
            return errors::InvalidArgument("Unknown output name: ", output_name);
        } else {
            *start = result->second.first;
            *stop = result->second.second;
            return Status::OK();
        }
    }

    void AsyncOpKernel::Compute(OpKernelContext* context) {
        Notification n;
        ComputeAsync(context, [&n]() { n.Notify(); });
        n.WaitForNotification();
    }

    // OpKernelConstruction ------------------------------------------------------

    OpKernelConstruction::OpKernelConstruction(
            DeviceType device_type, DeviceBase* device, Allocator* allocator,
            const NodeDef* node_def, const OpDef* op_def,
            const DataTypeSlice& input_types, const MemoryTypeSlice& input_memory_types,
            const DataTypeSlice& output_types,
            const MemoryTypeSlice& output_memory_types,
            Status* status)
        : device_type_(std::move(device_type)),
        device_(device),
        allocator_(allocator),
        def_(node_def),
        op_def_(op_def),
        input_types_(input_types),
        input_memory_types_(input_memory_types),
        output_types_(output_types),
        output_memory_types_(output_memory_types),
        status_(status) {}

    bool OpKernelConstruction::HasAttr(StringPiece attr_name) const {
        return HasNodeAttr(def(), attr_name);
    }

    void OpKernelConstruction::SetStatus(const Status& status) {
        status_->Update(status);
    }

    Status OpKernelConstruction::allocate_temp(DataType type,
            const TensorShape& shape,
            Tensor* out_temp) {
        AllocationAttributes attr;
        attr.allocation_will_be_logged = true;
        Tensor new_temp(allocator_, type, shape, attr);

        if (!new_temp.IsInitialized()) {
            return errors::ResourceExhausted(
                    "OOM when allocating temporary tensor with shape", shape.DebugString());
        }
        if (LogMemory::IsEnabled()) {
            LogMemory::RecordTensorAllocation(
                    def_->name(), LogMemory::OP_KERNEL_CONSTRUCTION_STEP_ID, new_temp);
        }
        *out_temp = new_temp;
        return Status::OK();
    }


    // OpKernelContext -----------------------------------------------------------

    OpKernelContext::OpKernelContext(Params* params)
        : OpKernelContext(
                params, static_cast<int>(params->op_kernel->output_types().size())) {}

    OpKernelContext::OpKernelContext(Params* params, int num_outputs)
        : params_(params), outputs_(num_outputs) {
            if (params_->track_allocations) {
                tracking_state_ = absl::make_unique<TrackingState>();
            }

            // params_->ensure_eigen_gpu_device();
            // if (params_->eigen_gpu_device != nullptr) {
            // Allocator* eigen_gpu_allocator = get_allocator(AllocatorAttributes());
            // Status s = params_->device->ReinitializeGpuDevice(
            // this, params_->eigen_gpu_device, params_->op_device_context,
            // eigen_gpu_allocator);
            // if (!s.ok()) {
            // SetStatus(s);
            // }
            // }
        }

    OpKernelContext::~OpKernelContext() {
        for (TensorValue& value : outputs_) {
            if (!value.is_ref()) {
                delete value.tensor;
            }
        }
        if (params_->track_allocations &&
                !tracking_state_->wrapped_allocators.empty()) {
            LOG(WARNING) << "OpKernelContext is tracking allocations but they are not "
                << "being consumed by the StepStatsCollector.";
            for (auto& wrapped_allocator : tracking_state_->wrapped_allocators) {
                wrapped_allocator.second->GetRecordsAndUnRef();
            }
        }
    }

    Allocator* OpKernelContext::get_allocator(AllocatorAttributes attr) {
        Allocator* allocator = nullptr;
        allocator = params_->device->GetAllocator(attr);

        if (TF_PREDICT_FALSE(track_allocations())) {
            DCHECK(tracking_state_);
            mutex_lock lock(tracking_state_->mu);
            for (const auto& wrapped : tracking_state_->wrapped_allocators) {
                if (wrapped.first == allocator) {
                    return wrapped.second;
                }
            }
            TrackingAllocator* wrapped_allocator =
                new TrackingAllocator(allocator, params_->track_allocations);
            tracking_state_->wrapped_allocators.push_back(
                    std::make_pair(allocator, wrapped_allocator));
            return wrapped_allocator;
        } else {
            return allocator;
        }
    }

    void OpKernelContext::SetStatus(const Status& status) {
        status_.Update(status);
    }

    Status OpKernelContext::input(StringPiece name, const Tensor** tensor) {
        int start, stop;
        TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
        if (stop != start + 1) {
            return errors::InvalidArgument("OpKernel used list-valued input name '",
                    name,
                    "' when single-valued input was "
                    "expected");
        }
        if (input_is_ref(start)) {
            return errors::InvalidArgument("OpKernel used ref input name '", name,
                    "' when non-ref input was expected");
        }
        *tensor = (*params_->inputs)[start].tensor;
        record_tensor_reference(**tensor);
        return Status::OK();
    }

    Status OpKernelContext::input_dtype(StringPiece name, DataType* dtype) const {
        int start, stop;
        TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
        if (stop != start + 1) {
            return errors::InvalidArgument("OpKernel used list-valued input name '",
                    name,
                    "' when single-valued input was "
                    "expected");
        }
        const TensorValue& value((*params_->inputs)[start]);
        *dtype = value.dtype();
        return Status::OK();
    }

    Status OpKernelContext::input_ref_mutex(StringPiece name, mutex** out_mutex) {
        int start, stop;
        TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
        if (stop != start + 1) {
            return errors::InvalidArgument("OpKernel used list-valued input name '",
                    name,
                    "' when single-valued input was expected");
        }
        *out_mutex = input_ref_mutex(start);
        return Status::OK();
    }

    const Tensor& OpKernelContext::input(int index) {
        CHECK_GE(index, 0);
        CHECK_LT(index, num_inputs()) << " name: " << op_kernel().name();
        CHECK(!input_is_ref(index));
        const Tensor& tensor = *((*params_->inputs)[index].tensor);
        // record_tensor_reference(tensor);
        return tensor;
    }
    Tensor OpKernelContext::mutable_input(int index, bool lock_held) {
        CHECK_GE(index, 0);
        CHECK_LT(index, num_inputs());
        CHECK(input_is_ref(index));
        // return a copy of the Ref acquired while holding the mutex
        if (lock_held) {
            Tensor& tensor = *((*params_->inputs)[index].tensor);
            record_tensor_reference(tensor);
            return tensor;
        } else {
            tf_shared_lock l(*input_ref_mutex(index));
            Tensor& tensor = *((*params_->inputs)[index].tensor);
            record_tensor_reference(tensor);
            return tensor;
        }
    }
    Status OpKernelContext::mutable_input(StringPiece name, Tensor* tensor,
            bool lock_held) {
        int start, stop;
        TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
        if (stop != start + 1) {
            return errors::InvalidArgument("OpKernel used list-valued input name '",
                    name,
                    "' when single-valued input was expected");
        }
        if (!input_is_ref(start)) {
            return errors::InvalidArgument("OpKernel used non-ref input name '", name,
                    "' when ref input was expected");
        }
        // return a copy of the Ref acquired while holding the mutex
        if (lock_held) {
            *tensor = *(*params_->inputs)[start].tensor;
        } else {
            tf_shared_lock l(*input_ref_mutex(start));
            *tensor = *(*params_->inputs)[start].tensor;
        }
        record_tensor_reference(*tensor);
        return Status::OK();
    }
    Status OpKernelContext::input_list(StringPiece name, OpInputList* list) {
        int start, stop;
        TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
        *list = OpInputList(this, start, stop);
        return Status::OK();
    }

    Status OpKernelContext::mutable_input_list(StringPiece name,
            OpMutableInputList* list) {
        int start, stop;
        TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
        *list = OpMutableInputList(this, start, stop);
        return Status::OK();
    }


    Status OpKernelContext::allocate_output(int index, const TensorShape& shape,
            Tensor** tensor) {
        if (index < 0) {
            return errors::Internal("allocate_output with bad index=", index,
                    " kernel=", params_->op_kernel->name());
        }
        if (index >= num_outputs()) {
            return errors::Internal("allocate_output with bad index=", index,
                    " num_outputs=", num_outputs(),
                    " kernel=", params_->op_kernel->name());
        }
        // bool forward_expected =
        // (params_->forward_from_array != nullptr && index >= 0 &&
        // params_->forward_from_array[index] >= 0);
        // if (forward_expected) {
        // return errors::Internal(
        // "Explicit allocate_output call where input forwarding required.  Try "
        // "turning off the ScopedAllocator optimizer.");
        // }
        AllocatorAttributes attr = output_alloc_attr(index);
        return allocate_output(index, shape, tensor, attr);
    }

    Status OpKernelContext::allocate_output(StringPiece name,
            const TensorShape& shape,
            Tensor** tensor) {
        int start, stop;
        TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
        if (stop != start + 1) {
            return errors::InvalidArgument("OpKernel used list-valued output name '",
                    name,
                    "' when single-valued output was "
                    "expected");
        }
        return allocate_output(start, shape, tensor);
    }
    Status OpKernelContext::allocate_output(StringPiece name,
            const TensorShape& shape,
            Tensor** tensor,
            AllocatorAttributes attr) {
        int start, stop;
        TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
        if (stop != start + 1) {
            return errors::InvalidArgument("OpKernel used list-valued output name '",
                    name,
                    "' when single-valued output was "
                    "expected");
        }
        return allocate_output(start, shape, tensor, attr);
    }

    Status OpKernelContext::allocate_tensor(
            DataType type, const TensorShape& shape, Tensor* out_tensor,
            AllocatorAttributes attr, const AllocationAttributes& allocation_attr) {
        Allocator* a = get_allocator(attr);
        // MEMDEBUG_CACHE_OP(op_kernel().name().c_str());
        // MEMDEBUG_CACHE_STEPID(step_id());
        Tensor new_tensor(a, type, shape,
                AllocationAttributes(allocation_attr.no_retry_on_failure,
                    /* allocation_will_be_logged= */ true,
                    allocation_attr.freed_by_func));

        if (!new_tensor.IsInitialized()) {
            return errors::ResourceExhausted(
                    "OOM when allocating tensor with shape", shape.DebugString(),
                    " and type ", DataTypeString(type), " on ", params_->device->name(),
                    " by allocator ", a->Name());
        }
        if (params_->log_memory) {
            LogMemory::RecordTensorAllocation(params_->op_kernel->name(),
                    params_->step_id, new_tensor);
        }
        record_tensor_reference(new_tensor);
        *out_tensor = std::move(new_tensor);
        return Status::OK();
    }

    Status OpKernelContext::allocate_output(int index, const TensorShape& shape,
            Tensor** output,
            AllocatorAttributes attr) {
        if (index < 0) {
            return errors::Internal("allocate_output with bad index=", index,
                    " kernel=", params_->op_kernel->name());
        }
        if (index >= num_outputs()) {
            return errors::Internal("allocate_output with bad index=", index,
                    " num_outputs=", outputs_.size(),
                    " kernel=", params_->op_kernel->name());
        }
        const DataType type = params_->op_kernel->output_type(index);
        if (IsRefType(type)) {
            return errors::Internal("allocate_output with ref type. index=", index,
                    " type=", type,
                    " kernel=", params_->op_kernel->name());
        }
        if (mutable_output(index) != nullptr) {
            return errors::Internal("allocate_output on same index multiple times.",
                    " index = ", index,
                    " mutable_output(index) = ", mutable_output(index),
                    " kernel=", params_->op_kernel->name());
        }
        // if (attr.scope_id > 0) {
        // maybe_initialize_scope_id_set();
        // if (!allocated_scope_ids_->insert(attr.scope_id).second) {
        // return errors::Internal(
        // "OpKernel ", params_->op_kernel->name(),
        // " called allocate_output at index ", index, " with scope_id ",
        // attr.scope_id,
        // " more than once.  Try turning off the ScopedAllocator optimizer.");
        // }
        // }
        auto output_tensor = MakeUnique<Tensor>();
        Status s = allocate_tensor(type, shape, output_tensor.get(), attr);
        if (s.ok()) {
            outputs_[index] = TensorValue(output_tensor.release());
            *output = outputs_[index].tensor;
        }
        return s;
    }

    Status OpKernelContext::allocate_temp(
            DataType type, const TensorShape& shape, Tensor* out_temp,
            AllocatorAttributes allocator_attr,
            const AllocationAttributes& allocation_attr) {
        if (allocator_attr.scope_id > 0) {
            // We do not allow ScopedAllocator calls from allocate_temp.  Unlike
            // allocate_persistent where we return an error if a kernel provides a
            // meaningful scope_id, here we clear the scope_id and return a temporary
            // buffer.  This is because it is legal for a kernel to call allocate_temp
            // and then set_output with the temp tensor.
            //
            // We achieve memory correctness by forcing an allocation in set_output and
            // copying over the tensor from the temp buffer.  Kernels which would like
            // to avoid this performance penalty should switch to calling
            // allocate_output.
            VLOG(2) << "Warning: OpKernel " << params_->op_kernel->name()
                << " called allocate_temp with scope_id " << allocator_attr.scope_id
                << ".  Switch to allocate_output to avoid performance penalty.";
            allocator_attr.scope_id = -1;
        }
        Status s =
            allocate_tensor(type, shape, out_temp, allocator_attr, allocation_attr);
        if (track_allocations() && s.ok() && out_temp->TotalBytes() > 0) {
            Allocator* a = get_allocator(allocator_attr);
            if (a->TracksAllocationSizes()) {
                int64 alloc_size = a->AllocatedSize(out_temp->tensor_data().data());
                record_temp_memory_allocation(alloc_size, *out_temp);
            }
        } else if (record_memory_consumption_) {
            DCHECK(tracking_state_);
            mutex_lock l(tracking_state_->stats_mu);
            tracking_state_->temp_memory_allocated += out_temp->TotalBytes();
        }
        return s;
    }



    Status OpKernelContext::set_output(StringPiece name, const Tensor& tensor) {
        int start, stop;
        TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
        if (stop != start + 1) {
            return errors::InvalidArgument("OpKernel used list-valued output name '",
                    name,
                    "' when single-valued output was "
                    "expected");
        }
        set_output(start, tensor);
        return Status::OK();
    }

    void OpKernelContext::set_output(int index, const Tensor& tensor) {
        CHECK_GE(index, 0);
        CHECK_LT(index, outputs_.size());
        const DataType type = params_->op_kernel->output_type(index);
        CHECK(!IsRefType(type));
        CHECK_EQ(mutable_output(index), nullptr);

        bool allocate_and_copy = false;
        if(allocate_and_copy){
        }else{
            outputs_[index] = TensorValue(new Tensor(tensor));
            if(track_allocations() && tensor.TotalBytes()>0){
                // track allocation
            }
        }
    }

    void OpKernelContext::set_output_ref(int index, mutex* mu,
            Tensor* tensor_for_ref) {
        CHECK_GE(index, 0);
        CHECK_LT(index, outputs_.size());
        CHECK(IsRefType(params_->op_kernel->output_type(index)));
        record_tensor_reference(*tensor_for_ref);
        outputs_[index] = TensorValue(mu, tensor_for_ref);
    }

    Status OpKernelContext::set_output_ref(StringPiece name, mutex* mu,
            Tensor* tensor_for_ref) {
        int start, stop;
        TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
        if (stop != start + 1) {
            return errors::InvalidArgument("OpKernel used list-valued output name '",
                    name,
                    "' when single-valued output was "
                    "expected");
        }
        set_output_ref(start, mu, tensor_for_ref);
        return Status::OK();
    }

    Status OpKernelContext::mutable_output(StringPiece name, Tensor** tensor) {
        int start, stop;
        TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
        if (stop != start + 1) {
            return errors::InvalidArgument("OpKernel used list-valued output name '",
                    name,
                    "' when single-valued output was "
                    "expected");
        }
        *tensor = mutable_output(start);
        return Status::OK();
    }

    void OpKernelContext::record_temp_memory_allocation(int64 size,
            const Tensor& t) {
        if (tracking_state_) {
            mutex_lock l(tracking_state_->stats_mu);
            tracking_state_->temp_memory_allocated += size;
            tracking_state_->temp_tensor_buffer_and_size.emplace_back(
                    static_cast<const void*>(t.tensor_data().data()), size);
        }
    }
    int64 OpKernelContext::temp_memory_allocated() const {
        if (tracking_state_) {
            mutex_lock l(tracking_state_->stats_mu);
            return tracking_state_->temp_memory_allocated;
        } else {
            return 0;
        }
    }

    void OpKernelContext::record_persistent_memory_allocation(int64 size,
            int64 alloc_id) {
        if (tracking_state_) {
            mutex_lock l(tracking_state_->stats_mu);
            tracking_state_->persistent_memory_allocated += size;
            if (alloc_id >= 0) {
                tracking_state_->persistent_alloc_ids.push_back(alloc_id);
            }
        }
    }

    int64 OpKernelContext::persistent_memory_allocated() const {
        if (tracking_state_) {
            mutex_lock l(tracking_state_->stats_mu);
            return tracking_state_->persistent_memory_allocated;
        } else {
            return 0;
        }
    }

    std::vector<int64> OpKernelContext::persistent_alloc_ids() const {
        if (tracking_state_) {
            mutex_lock l(tracking_state_->stats_mu);
            return std::vector<int64>(tracking_state_->persistent_alloc_ids.begin(),
                    tracking_state_->persistent_alloc_ids.end());
        } else {
            return std::vector<int64>();
        }
    }

    void OpKernelContext::clear_recorded_memory() {
        if (tracking_state_) {
            mutex_lock l(tracking_state_->stats_mu);
            tracking_state_->temp_memory_allocated = 0;
            tracking_state_->persistent_memory_allocated = 0;
            tracking_state_->temp_tensor_buffer_and_size.clear();
            tracking_state_->persistent_alloc_ids.clear();
        }
    }

    void OpKernelContext::set_record_memory_consumption(bool v) {
        record_memory_consumption_ = v;
        if (v && !tracking_state_) {
            tracking_state_ = absl::make_unique<TrackingState>();
        }
    }
    template <>
        const Eigen::ThreadPoolDevice& OpKernelContext::eigen_device() const {
            return eigen_cpu_device();
        }

    // OpKernel registration ------------------------------------------------------

    struct KernelRegistration {
        KernelRegistration(const KernelDef& d, StringPiece c,
                std::unique_ptr<kernel_factory::OpKernelFactory> f)
            : def(d), kernel_class_name(c), factory(std::move(f)) {}

        const KernelDef def;
        const string kernel_class_name;
        std::unique_ptr<kernel_factory::OpKernelFactory> factory;
    };


    // This maps from 'op_type' + DeviceType to the set of KernelDefs and
    // factory functions for instantiating the OpKernel that matches the
    // KernelDef.
    struct KernelRegistry {
        mutex mu;
        std::unordered_multimap<string, KernelRegistration> registry GUARDED_BY(mu);
    };

    void LoadDynamicKernelsInternal() {
        // load so library
    }

    // Mechanism for loading existing kernel libraries.
    void LoadDynamicKernels() {
        // TODO(gunan): As more features are available, add intelligent kernel
        // selection, and dropping unsuitable kernel logic here.
        static std::once_flag dll_loader_flag;
        std::call_once(dll_loader_flag, LoadDynamicKernelsInternal);
    }

    void* GlobalKernelRegistry() {
        static KernelRegistry* global_kernel_registry = new KernelRegistry;
        return global_kernel_registry;
    }

    static KernelRegistry* GlobalKernelRegistryTyped() {
#ifdef AUTOLOAD_DYNAMIC_KERNELS
        LoadDynamicKernels();
#endif  // AUTOLOAD_DYNAMIC_KERNELS
        return reinterpret_cast<KernelRegistry*>(GlobalKernelRegistry());
    }

    static string Key(StringPiece op_type, const DeviceType& device_type,
            StringPiece label) {
        return strings::StrCat(op_type, ":", DeviceTypeString(device_type), ":",
                label);
    }

    namespace kernel_factory {

        void OpKernelRegistrar::InitInternal(const KernelDef* kernel_def,
                StringPiece kernel_class_name,
                std::unique_ptr<OpKernelFactory> factory) {
            // See comments in register_kernel::Name in header for info on _no_register.
            if (kernel_def->op() != "_no_register") {
                const string key =
                    Key(kernel_def->op(), DeviceType(kernel_def->device_type()),
                            kernel_def->label());

                // To avoid calling LoadDynamicKernels DO NOT CALL GlobalKernelRegistryTyped
                // here.
                // InitInternal gets called by static initializers, so it ends up executing
                // before main. This causes LoadKernelLibraries function to get called
                // before some file libraries can initialize, which in turn crashes the
                // program flakily. Until we get rid of static initializers in kernel
                // registration mechanism, we have this workaround here.
                auto global_registry =
                    reinterpret_cast<KernelRegistry*>(GlobalKernelRegistry());
                mutex_lock l(global_registry->mu);
                global_registry->registry.emplace(
                        key,
                        KernelRegistration(*kernel_def, kernel_class_name, std::move(factory)));
            }
            delete kernel_def;
        }

        OpKernel* OpKernelRegistrar::PtrOpKernelFactory::Create(
                OpKernelConstruction* context) {
            return (*create_func_)(context);
        }

    }  // namespace kernel_factory

    namespace {

        static const StringPiece kKernelAttr("_kernel");

        // TODO(irving): Replace with const Node& version below.
        Status FindKernelRegistration(
                const DeviceType& device_type, StringPiece node_name,
                bool has_experimental_debug_info,
                const NodeDef_ExperimentalDebugInfo& experimental_debug_info,
                StringPiece node_op, AttrSlice node_attrs, const KernelRegistration** reg,
                bool* was_attr_mismatch) {
            *reg = nullptr;
            *was_attr_mismatch = false;
            // Label defaults to empty if not found in NodeDef.
            // const string& label = GetNodeAttrString(node_attrs, kKernelAttr);
            const string& label = "";

            const string key = Key(node_op, device_type, label);
            auto typed_registry = GlobalKernelRegistryTyped();
            tf_shared_lock lock(typed_registry->mu);
            auto regs = typed_registry->registry.equal_range(key);
            for (auto iter = regs.first; iter != regs.second; ++iter) {
                //check each match kernel, determine use which one
                bool match;
                TF_RETURN_IF_ERROR(KernelAttrsMatch(iter->second.def, node_attrs, &match));
                if(match){
                    if(*reg!=nullptr){
                        // if multiple matched, find the one that has the max prior
                        if ((*reg)->def.priority() == iter->second.def.priority()) {
                            return errors::InvalidArgument(
                                    "Multiple OpKernel registrations match NodeDef at the same "
                                    "priority '",
                                    node_name,
                                    "': '", (*reg)->def.ShortDebugString(), "' and '",
                                    iter->second.def.ShortDebugString(), "'");
                        } else if ((*reg)->def.priority() > iter->second.def.priority()) {
                            continue;
                        }
                    }
                    *reg = &iter->second;
                }
            }
            // reg can be nullptr

            return Status::OK();
        }

        Status FindKernelRegistration(const DeviceType& device_type,
                const NodeDef& node_def,
                const KernelRegistration** reg,
                bool* was_attr_mismatch) {
            return FindKernelRegistration(
                    device_type, node_def.name(), node_def.has_experimental_debug_info(),
                    node_def.experimental_debug_info(), node_def.op(),
                    AttrSlice(node_def), reg, was_attr_mismatch);
        }
    }// namespace

    std::unique_ptr<OpKernel> CreateOpKernel(
            DeviceType device_type, DeviceBase* device, Allocator* allocator,
            const NodeDef& node_def, Status* status) {
        OpKernel* kernel = nullptr;
        *status = CreateOpKernel(std::move(device_type), device, allocator, node_def, &kernel);
        return std::unique_ptr<OpKernel>(kernel);
    }

    KernelList GetAllRegisteredKernels() {
        return GetFilteredRegisteredKernels([](const KernelDef& k) { return true; });
    }
    void LogAllRegisteredKernels() {
        KernelList kernel_list = GetAllRegisteredKernels();
        for (const auto& kernel_def : kernel_list.kernel()) {
            LOG(INFO) << "OpKernel ('" << kernel_def.ShortDebugString() << "')";
        }
    }

    KernelList GetFilteredRegisteredKernels(
            const std::function<bool(const KernelDef&)>& predicate) {
        KernelRegistry* const typed_registry = GlobalKernelRegistryTyped();
        KernelList kernel_list;
        tf_shared_lock lock(typed_registry->mu);
        kernel_list.mutable_kernel()->Reserve(typed_registry->registry.size());
        for (const auto& p : typed_registry->registry) {
            const KernelDef& kernel_def = p.second.def;
            if (predicate(kernel_def)) {
                *kernel_list.add_kernel() = kernel_def;
            }
        }
        return kernel_list;
    }

    KernelList GetRegisteredKernelsForOp(StringPiece op_name) {
        auto op_pred = [op_name](const KernelDef& k) { return k.op() == op_name; };
        return GetFilteredRegisteredKernels(op_pred);
    }

    string KernelsRegisteredForOp(StringPiece op_name) {
        KernelList kernel_list = GetRegisteredKernelsForOp(op_name);
        if (kernel_list.kernel_size() == 0) return "  <no registered kernels>\n";
        string ret;
        for (const auto& kernel_def : kernel_list.kernel()) {
            strings::StrAppend(&ret, "  device='", kernel_def.device_type(), "'");
            if (!kernel_def.label().empty()) {
                strings::StrAppend(&ret, "; label='", kernel_def.label(), "'");
            }
            for (int i = 0; i < kernel_def.constraint_size(); ++i) {
                strings::StrAppend(
                        &ret, "; ", kernel_def.constraint(i).name(), " in ",
                        SummarizeAttrValue(kernel_def.constraint(i).allowed_values()));
            }
            strings::StrAppend(&ret, "\n");
        }
        return ret;
    }

    Status CreateOpKernel(DeviceType device_type, DeviceBase* device,
            Allocator* allocator,
            const NodeDef& node_def,
            OpKernel** kernel) {
        VLOG(1) << "Instantiating kernel for node: " << SummarizeNodeDef(node_def);

        // Look up the Op registered for this op name.
        const OpDef* op_def = nullptr;
        TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUpOpDef(node_def.op(), &op_def));

        // Validate node_def against OpDef.
        TF_RETURN_IF_ERROR(ValidateNodeDef(node_def, *op_def));

        // Look up kernel registration.
        const KernelRegistration* registration;
        bool was_attr_mismatch;
        Status s = FindKernelRegistration(device_type, node_def, &registration,
                &was_attr_mismatch);
        if (!s.ok()) {
            errors::AppendToMessage(&s, " when instantiating ", node_def.op());
            return s;
        }
        if (registration == nullptr) {
            s.Update(errors::NotFound("No registered '", node_def.op(),
                        "' OpKernel for '", DeviceTypeString(device_type),
                        "' devices compatible with node ",
                        FormatNodeDefForError(node_def)));
            if (was_attr_mismatch) {
                errors::AppendToMessage(
                        &s, " (OpKernel was found, but attributes didn't match) ",
                        "Requested Attributes: ", SummarizeAttrs(node_def));
            }
            errors::AppendToMessage(
                    &s, ".  Registered:", KernelsRegisteredForOp(node_def.op()));
            return s;
        }

        // Get signature from the OpDef & NodeDef
        DataTypeVector inputs;
        DataTypeVector outputs;
        s.Update(InOutTypesForNode(node_def, *op_def, &inputs, &outputs));
        if (!s.ok()) {
            errors::AppendToMessage(&s, " for node: ", FormatNodeDefForError(node_def));
            return s;
        }

        // We are creating a kernel for an op registered in
        // OpRegistry::Global(), we consult the kernel registry to decide
        // the kernel's input and output memory types.
        MemoryTypeVector input_memory_types;
        MemoryTypeVector output_memory_types;
        // TF_RETURN_IF_ERROR(MemoryTypesForNode(OpRegistry::Global(), device_type,
        // node_def, &input_memory_types,
        // &output_memory_types));

        // Everything needed for OpKernel construction.
        OpKernelConstruction context(
                device_type, device, allocator, &node_def, op_def, inputs,
                input_memory_types, outputs, output_memory_types, &s);
        *kernel = registration->factory->Create(&context);
        if (!s.ok()) {
            delete *kernel;
            *kernel = nullptr;
        }
        return s;
    }

    void OpKernelConstruction::CtxFailure(const Status& s) {
        VLOG(1) << s;
        SetStatus(s);
    }

    void OpKernelConstruction::CtxFailureWithWarning(const Status& s) {
        LOG(WARNING) << s;
        SetStatus(s);
    }

    void OpKernelConstruction::CtxFailure(const char* file, int line,
            const Status& s) {
        VLOG(1) << "OP_REQUIRES failed at " << io::Basename(file) << ":" << line
            << " : " << s;
        SetStatus(s);
    }

    void OpKernelConstruction::CtxFailureWithWarning(const char* file, int line,
            const Status& s) {
        LOG(WARNING) << "OP_REQUIRES failed at " << io::Basename(file) << ":" << line
            << " : " << s;
        SetStatus(s);
    }

    void OpKernelContext::CtxFailure(const Status& s) {
        VLOG(1) << s;
        SetStatus(s);
    }

    void OpKernelContext::CtxFailureWithWarning(const Status& s) {
        LOG(WARNING) << s;
        SetStatus(s);
    }

    void OpKernelContext::CtxFailure(const char* file, int line, const Status& s) {
        VLOG(1) << "OP_REQUIRES failed at " << io::Basename(file) << ":" << line
            << " : " << s;
        SetStatus(s);
    }

    void OpKernelContext::CtxFailureWithWarning(const char* file, int line,
            const Status& s) {
        LOG(WARNING) << "OP_REQUIRES failed at " << io::Basename(file) << ":" << line
            << " : " << s;
        SetStatus(s);
    }

    void CheckNotInComputeAsync(OpKernelContext* ctx,
            const char* correct_macro_name) {
        CHECK_EQ(nullptr, ctx->op_kernel().AsAsync())
            << "Use " << correct_macro_name << " in AsyncOpKernel implementations.";
    }
}// namespace dlxnet
