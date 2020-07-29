#include "dlxnet/core/common_runtime/copy_tensor.h"


namespace dlxnet{
    namespace {
        void CopyHostToDevice(const Tensor* input, Allocator* cpu_allocator,
                Allocator* out_allocator, StringPiece edge_name,
                Device* dst, Tensor* output,
                DeviceContext* recv_dev_context, StatusCallback done,
                bool sync_dst_compute) {
            recv_dev_context->CopyCPUTensorToDevice(input, dst, output, std::move(done),
                    sync_dst_compute);
        }
    }
    // static
    void CopyTensor::ViaDMA(StringPiece edge_name, DeviceContext* send_dev_context,
            DeviceContext* recv_dev_context, Device* src,
            Device* dst, const AllocatorAttributes src_alloc_attr,
            const AllocatorAttributes dst_alloc_attr,
            const Tensor* input, Tensor* output,
            int dev_to_dev_stream_index, StatusCallback done,
            bool sync_dst_compute) {
        VLOG(1) << "Copy " << edge_name;

        const DeviceType src_device_type(
                src_alloc_attr.on_host() ? DEVICE_CPU : src->attributes().device_type());
        const DeviceType dst_device_type(
                dst_alloc_attr.on_host() ? DEVICE_CPU : dst->attributes().device_type());
        const bool non_cpu_src = src_device_type != DeviceType(DEVICE_CPU);
        const bool non_cpu_dst = dst_device_type != DeviceType(DEVICE_CPU);

        // TODO(phawkins): choose an allocator optimal for both the src and dst
        // devices, not just the src device.
        AllocatorAttributes host_alloc_attrs;
        host_alloc_attrs.set_gpu_compatible(true);
        host_alloc_attrs.set_on_host(true);
        Allocator* cpu_allocator = src->GetAllocator(host_alloc_attrs);
        Allocator* out_allocator = dst->GetAllocator(dst_alloc_attr);

        // E.g., gpu -> gpu
        if (non_cpu_src && non_cpu_dst) {
            return ;
        }

        // E.g., gpu -> cpu
        if (non_cpu_src && !non_cpu_dst) {
            // Device to host copy.
            CopyDeviceToHost(input, cpu_allocator, out_allocator, edge_name, src,
                    output, send_dev_context, std::move(done));
            return;
        }

        // E.g., cpu -> gpu
        if (!non_cpu_src && non_cpu_dst) {
            // Host to Device copy.
            CopyHostToDevice(input, cpu_allocator, out_allocator, edge_name, dst,
                    output, recv_dev_context, std::move(done),
                    sync_dst_compute);
            return;
        }
        // cpu -> cpu
        CHECK(!non_cpu_src && !non_cpu_dst);
        *output = *input;
        done(Status::OK());
    }

    void CopyDeviceToHost(const Tensor* input, Allocator* cpu_allocator,
            Allocator* out_allocator, StringPiece edge_name,
            Device* src, Tensor* output,
            DeviceContext* send_dev_context, StatusCallback done) {
        send_dev_context->CopyDeviceTensorToCPU(input, edge_name, src, output,
                std::move(done));
    }
} // namespace dlxnet
