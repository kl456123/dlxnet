#ifndef DLXNET_CORE_COMMON_RUNTIME_COPY_TENSOR_H_
#define DLXNET_CORE_COMMON_RUNTIME_COPY_TENSOR_H_
#include "dlxnet/core/common_runtime/device.h"
#include "dlxnet/core/framework/allocator.h"
#include "dlxnet/core/framework/device_base.h"
#include "dlxnet/core/framework/tensor.h"
#include "dlxnet/core/framework/types.h"
#include "dlxnet/core/lib/core/status.h"
#include "dlxnet/core/platform/types.h"

namespace dlxnet{
    class CopyTensor {
        public:
            typedef void (*CopyFunction)(
                    DeviceContext* send_dev_context, DeviceContext* recv_dev_context,
                    Device* src, Device* dst, const AllocatorAttributes src_alloc_attr,
                    const AllocatorAttributes dst_alloc_attr, const Tensor* input,
                    Tensor* output, int dev_to_dev_stream_index, StatusCallback done);

            // Copies "input" to "output" between devices accessible to the
            // local process via some DMA-like method.  "edge_name" is the name
            // of the tensor being copied, for debugging purposes. Depending on
            // the type of devices and memory in use, the copy may be performed
            // synchronously or asynchronously.  'done' will be invoked only
            // after the copy is actually complete.
            static void ViaDMA(StringPiece edge_name, DeviceContext* send_dev_context,
                    DeviceContext* recv_dev_context, Device* src, Device* dst,
                    const AllocatorAttributes src_alloc_attr,
                    const AllocatorAttributes dst_alloc_attr,
                    const Tensor* input, Tensor* output,
                    int dev_to_dev_stream_index, StatusCallback done,
                    bool sync_dst_compute = true);
            // Object used to call Register() at static-initialization time.
            // Note: This should only ever be used as a global-static object; no stack
            // or heap instances.
            class Registration {
                public:
                    Registration(DeviceType sender_device_type, DeviceType receiver_device_type,
                            CopyFunction copy_function) {
                        TF_QCHECK_OK(
                                Register(sender_device_type, receiver_device_type, copy_function));
                    }
            };

        private:
            // Register a function for copying between two specific DeviceTypes.
            // Note: This should only be called via the constructor of
            // CopyTensor::Registration.
            static Status Register(DeviceType sender_device_type,
                    DeviceType receiver_device_type,
                    CopyFunction copy_function);
    };

    void CopyDeviceToHost(const Tensor* input, Allocator* cpu_allocator,
            Allocator* out_allocator, StringPiece edge_name,
            Device* src, Tensor* output,
            DeviceContext* send_dev_context, StatusCallback done);
} // namespace dlxnet



#endif
