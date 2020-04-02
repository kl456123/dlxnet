#ifndef DLXNET_CORE_COMMON_RUNTIME_THREADPOOL_DEVICE_H_
#define DLXNET_CORE_COMMON_RUNTIME_THREADPOOL_DEVICE_H_
#include "dlxnet/core/common_runtime/local_device.h"
#include "dlxnet/core/public/session_options.h"

namespace dlxnet{

    class  ThreadPoolDevice : public LocalDevice{
        public:
            ThreadPoolDevice(const SessionOptions& options, const string& name,
                    Bytes memory_limit, const DeviceLocality& locality,
                    Allocator* allocator);
            ~ThreadPoolDevice()override;

            Allocator* GetAllocator(AllocatorAttributes /*attr*/)override;
            Status Sync()override{
                return Status::OK();
            }

            Status MakeTensorFromProto(const TensorProto& tensor_proto,
                    const AllocatorAttributes alloc_attrs,
                    Tensor* tensor) override;
        private:
            Allocator* allocator_; // not owned, cpu allocator always exist in whole lifetime

    };
}


#endif
