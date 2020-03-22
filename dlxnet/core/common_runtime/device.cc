#include "dlxnet/core/common_runtime/device.h"
#include "dlxnet/core/lib/random/random.h"


namespace dlxnet{
    Device::Device(Env* env, const DeviceAttributes& device_attributes)
        : DeviceBase(env), device_attributes_(device_attributes) {
            CHECK(DeviceNameUtils::ParseFullName(name(), &parsed_name_))
            << "Invalid device name: " << name();
            // rmgr_ = new ResourceMgr(parsed_name_.job);
        }

    Device::~Device() {
        // if (rmgr_ != nullptr) {
        // DeleteResourceMgr();
        // }
    }
    void Device::Sync(const DoneCallback& done) { done(Sync()); }
    // static
    DeviceAttributes Device::BuildDeviceAttributes(
            const string& name, DeviceType device, Bytes memory_limit,
            const DeviceLocality& locality, const string& physical_device_desc) {
        DeviceAttributes da;
        da.set_name(name);
        do {
            da.set_incarnation(random::New64());
        } while (da.incarnation() == 0);  // This proto field must not be zero
        da.set_device_type(device.type());
        da.set_memory_limit(memory_limit);
        *da.mutable_locality() = locality;
        da.set_physical_device_desc(physical_device_desc);
        return da;
    }
}
