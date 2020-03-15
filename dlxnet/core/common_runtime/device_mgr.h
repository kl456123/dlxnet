#ifndef DLXNET_CORE_COMMON_RUNTIME_DEVICE_MGR_H_
#define DLXNET_CORE_COMMON_RUNTIME_DEVICE_MGR_H_
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dlxnet/core/common_runtime/device.h"
#include "dlxnet/core/lib/status.h"
#include "dlxnet/core/lib/stringpiece.h"
#include "dlxnet/core/platform/macros.h"
#include "dlxnet/core/platform/types.h"
#include "dlxnet/core/lib/gtl/array_slice.h"
#include  "dlxnet/core/platform/hash.h"


namespace dlxnet{
    class DeviceAttributes;

    // Represents a set of devices.
    class DeviceMgr {
        public:
            DeviceMgr() = default;
            virtual ~DeviceMgr();

            // Returns attributes of all devices.
            virtual void ListDeviceAttributes(
                    std::vector<DeviceAttributes>* devices) const = 0;

            // Returns raw pointers to the underlying devices.
            virtual std::vector<Device*> ListDevices() const = 0;

            // Returns a string listing all devices.
            virtual string DebugString() const = 0;

            // Returns a string of all the device mapping.
            virtual string DeviceMappingString() const = 0;

            // Assigns *device with pointer to Device of the given name.
            // Accepts either a full device name, or just the replica-local suffix.
            virtual Status LookupDevice(StringPiece name, Device** device) const = 0;

            // Clears given containers of all devices if 'container' is
            // non-empty. Otherwise, clears default containers of all devices.
            virtual void ClearContainers(gtl::ArraySlice<string> containers) const = 0;

            virtual int NumDeviceType(const string& type) const = 0;

            TF_DISALLOW_COPY_AND_ASSIGN(DeviceMgr);
    };

    // Represents a static set of devices.
    class StaticDeviceMgr : public DeviceMgr {
        public:
            // Constructs a StaticDeviceMgr from a list of devices.
            explicit StaticDeviceMgr(std::vector<std::unique_ptr<Device>> devices);
            // Constructs a StaticDeviceMgr managing a single device.
            explicit StaticDeviceMgr(std::unique_ptr<Device> device);
            ~StaticDeviceMgr() override;
            void ListDeviceAttributes(
                    std::vector<DeviceAttributes>* devices) const override;
            std::vector<Device*> ListDevices() const override;
            string DebugString() const override;
            string DeviceMappingString() const override;
            Status LookupDevice(StringPiece name, Device** device) const override;
            void ClearContainers(gtl::ArraySlice<string> containers) const override;
            int NumDeviceType(const string& type) const override;
        private:
            std::unordered_map<StringPiece, Device*, StringPieceHasher> device_map_;
            const std::vector<std::unique_ptr<Device>> devices_;
            std::unordered_map<string, int> device_type_counts_;
            TF_DISALLOW_COPY_AND_ASSIGN(StaticDeviceMgr);
    };

}


#endif
