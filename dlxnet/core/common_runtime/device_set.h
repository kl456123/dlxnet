#ifndef DLXNET_CORE_COMMON_RUNTIME_DEVICE_SET_H_
#define DLXNET_CORE_COMMON_RUNTIME_DEVICE_SET_H_
#include <vector>
#include <unordered_map>

#include "dlxnet/core/platform/macros.h"
#include "dlxnet/core/platform/logging.h"
#include "dlxnet/core/common_runtime/device.h"

// container class, it doest not own anythings


namespace dlxnet{
    class DeviceSet{
        public:
            DeviceSet();
            ~DeviceSet();

            // Does not take ownership of 'device'.
            void AddDevice(Device* device);

            // Set the device designated as the "client".  This device
            // must also be registered via AddDevice().
            void set_client_device(Device* device) {
                DCHECK(client_device_ == nullptr);
                client_device_ = device;
            }
        private:
            // Not owned.
            std::vector<Device*> devices_;

            // Fullname -> device* for device in devices_.
            std::unordered_map<string, Device*> device_by_name_;

            // client_device_ points to an element of devices_ that we consider
            // to be the client device (in this local process).
            Device* client_device_ = nullptr;

            TF_DISALLOW_COPY_AND_ASSIGN(DeviceSet);
    };
}


#endif
