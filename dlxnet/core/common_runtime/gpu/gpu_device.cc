#include "dlxnet/core/common_runtime/gpu/gpu_device.h"
#include "dlxnet/core/common_runtime/gpu/gpu_id_manager.h"
#include "dlxnet/core/lib/status.h"
#include "dlxnet/core/lib/core/notification.h"
#include "dlxnet/core/platform/logging.h"
#include "dlxnet/core/common_runtime/gpu/gpu_util.h"


namespace dlxnet{
    BaseGPUDevice::BaseGPUDevice(const SessionOptions& options, const string& name,
            Bytes memory_limit, const DeviceLocality& locality,
            TfGpuId tf_gpu_id,
            const string& physical_device_desc,
            Allocator* gpu_allocator, Allocator* cpu_allocator,
            bool sync_every_op)
        : LocalDevice(options, Device::BuildDeviceAttributes(name, DEVICE_GPU,
                    memory_limit, locality,
                    physical_device_desc)),
        gpu_allocator_(gpu_allocator),
        cpu_allocator_(cpu_allocator),
        tf_gpu_id_(tf_gpu_id),
        sync_every_op_(sync_every_op) {
            // GPUProcessState::singleton()->EnableGPUDevice();
        }

    BaseGPUDevice::~BaseGPUDevice() {
        // delete gpu_device_info_;
        // gpu_allocator_->DeallocateRaw(scratch_);
        device_context_->Unref();
    }

    string BaseGPUDevice::ComputeOpKernelDebugString(const OpKernel& op_kernel,
            const int& stream_id) {
        return strings::StrCat(op_kernel.name(), " op ", op_kernel.type_string(),
                " on GPU ", tf_gpu_id_, " stream[", stream_id,
                "]");
    }

    Status BaseGPUDevice::Init(const SessionOptions& options) {
        device_context_ = new GPUDeviceContext(0);
        timestamped_allocator_ =
            options.config.gpu_options().experimental().timestamped_allocator();

        PlatformGpuId platform_gpu_id;
        TF_RETURN_IF_ERROR(
                GpuIdManager::TfToPlatformGpuId(tf_gpu_id_, &platform_gpu_id));

        return Status::OK();
    }

    void BaseGPUDevice::CopyTensorInSameDevice(const Tensor* input_tensor,
            Tensor* output_tensor,
            const DeviceContext* device_context,
            StatusCallback done) {
        // GPUUtil::CopyGPUTensorToSameGPU(static_cast<Device*>(this), device_context,
        // input_tensor, output_tensor, std::move(done));
    }

    // Based on the semantics of Device::Sync this call should wait for
    // all streams not just the current one.
    Status BaseGPUDevice::Sync() { return GPUUtil::SyncAll(this); }

    void BaseGPUDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
        // NOTE(tucker): We need to discriminate between Eigen GPU
        // operations and all others.  If an operation is Eigen
        // implemented (or otherwise tries to launch a GPU kernel
        // directly), we need to establish a stacked-scoped environment
        // that directs it to execute on the proper device.  Otherwise we
        // expect the Op to use StreamExecutor directly and correctly.
        GPUDeviceContext* gpu_device_context = device_context_;
        if (context->op_device_context() != nullptr) {
            gpu_device_context =
                static_cast<GPUDeviceContext*>(context->op_device_context());
        }
        const bool vlog_1 = VLOG_IS_ON(1);
        const auto stream_id = gpu_device_context->stream_id();
        if (vlog_1) {
            VLOG(1) << "GpuDevice::ComputeHelper "
                << ComputeOpKernelDebugString(*op_kernel, stream_id);
        }

        op_kernel->Compute(context);
        if (context->status().ok()) {
            if(vlog_1){
                VLOG(1) << "GpuDevice::ComputeHelper scheduled "
                    << ComputeOpKernelDebugString(*op_kernel, stream_id);
            }
        }else{
            if (vlog_1) {
                VLOG(1) << "GpuDevice::ComputeHelper failed to schedule "
                    << ComputeOpKernelDebugString(*op_kernel, stream_id);
            }
        }
    }
    void BaseGPUDevice::ComputeAsync(AsyncOpKernel* op_kernel,
            OpKernelContext* context,
            AsyncOpKernel::DoneCallback done) {
        GPUDeviceContext* gpu_device_context = device_context_;
        if (context->op_device_context() != nullptr) {
            gpu_device_context =
                static_cast<GPUDeviceContext*>(context->op_device_context());
        }
        // se::Stream* stream = gpu_device_context->stream();
        const auto stream_id = gpu_device_context->stream_id();

        VLOG(1) << "GpuDevice::ComputeAsync " << op_kernel->name() << " op "
            << op_kernel->type_string() << " on GPU" << tf_gpu_id_ << " stream["
            << stream_id << "]";

        op_kernel->ComputeAsync(context, std::move(done));
    }

    Status BaseGPUDevice::MaybeCopyTensorToGPU(
            const AllocatorAttributes& alloc_attrs, const Tensor& from, Tensor* to,
            StatusCallback done){
        if (alloc_attrs.on_host()) {
            *to = from;
            done(Status::OK());
            return Status::OK();
        } else {
            AllocationAttributes allocation_attr;
            auto* copy = new Tensor(GetAllocator(alloc_attrs), from.dtype(),
                    from.shape(), allocation_attr);
            // If the tensor is not initialized, we likely ran out of memory.
            if (!copy->IsInitialized()) {
                delete copy;
                Status err = errors::ResourceExhausted(
                        "OOM when allocating tensor of shape ", from.shape().DebugString(),
                        " and type ", DataTypeString(from.dtype()));
                done(err);
                return err;
            }
            auto wrapped_done = [to, copy, done = std::move(done)](const Status& s) {
                if (s.ok()) {
                    *to = std::move(*copy);
                }
                delete copy;
                done(s);
            };

            device_context_->CopyCPUTensorToDevice(
                    &from, this, copy, std::move(wrapped_done),
                    !timestamped_allocator_ /*sync_dst_compute*/);
            return Status::OK();
        }
    }

    Status BaseGPUDevice::MakeTensorFromProto(const TensorProto& tensor_proto,
            const AllocatorAttributes alloc_attrs,
            Tensor* tensor) {
        // first from disk to cpu memory
        AllocatorAttributes attr;
        attr.set_on_host(true);
        attr.set_gpu_compatible(true);
        Allocator* host_alloc = GetAllocator(attr);
        Tensor parsed(tensor_proto.dtype());
        if (!parsed.FromProto(host_alloc, tensor_proto)) {
            return errors::InvalidArgument("Cannot parse tensor from proto: ",
                    tensor_proto.DebugString());
        }
        if (parsed.dtype() == DT_VARIANT) {
        }else{
            Notification n;
            Status status;
            TF_RETURN_IF_ERROR(MaybeCopyTensorToGPU(alloc_attrs, parsed, tensor,
                        [&n, &status](const Status& s) {
                        status = s;
                        n.Notify();
                        }));
            n.WaitForNotification();
            return status;
        }
    }

    Status BaseGPUDeviceFactory::ListPhysicalDevices(std::vector<string>* devices) {
        return Status::OK();
    }


    // device factory
    Status BaseGPUDeviceFactory::CreateDevices(
            const SessionOptions& options, const string& name_prefix,
            std::vector<std::unique_ptr<Device>>* devices) {
    }
}// namespace dlxnet
