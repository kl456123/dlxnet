#include "dlxnet/core/common_runtime/threadpool_device.h"


namespace dlxnet{
    ThreadPoolDevice::ThreadPoolDevice(const SessionOptions& options,const string& name,
            Bytes memory_limit, const DeviceLocality& locality, Allocator* allocator)
        :Device(options.env, Device::BuildDeviceAttributes(
                    name, DEVICE_CPU, memory_limit, locality)),
        allocator_(allocator){
        }

    ThreadPoolDevice::~ThreadPoolDevice(){}
    Allocator* ThreadPoolDevice::GetAllocator(AllocatorAttributes attr){
        return allocator_;
    }
}
