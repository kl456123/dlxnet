#include "dlxnet/core/common_runtime/threadpool_device.h"


namespace dlxnet{
    ThreadPoolDevice::ThreadPoolDevice(const SessionOptions& options,const string& name,
            Bytes memory_limit, const DeviceLocality& locality, Allocator* allocator)
        :LocalDevice(options, Device::BuildDeviceAttributes(
                    name, DEVICE_CPU, memory_limit, locality)),
        allocator_(allocator){
        }

    Status ThreadPoolDevice::MakeTensorFromProto(
            const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs,
            Tensor* tensor) {
        if (tensor_proto.dtype() > 0 && tensor_proto.dtype() <= DataType_MAX) {
            Tensor parsed(tensor_proto.dtype());
            if (parsed.FromProto(allocator_, tensor_proto)) {
                *tensor = std::move(parsed);
                return Status::OK();
            }
        }
        return errors::InvalidArgument("Cannot parse tensor from proto: ",
                tensor_proto.DebugString());
    }

    ThreadPoolDevice::~ThreadPoolDevice(){}
    Allocator* ThreadPoolDevice::GetAllocator(AllocatorAttributes attr){
        return allocator_;
    }
}
