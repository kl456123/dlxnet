#include "dlxnet/core/common_runtime/device_set.h"
#include "dlxnet/core/util/device_name_utils.h"
#include "dlxnet/core/lib/gtl/map_util.h"
#include "dlxnet/core/common_runtime/device_factory.h"


namespace dlxnet{
    DeviceSet::DeviceSet() {}

    DeviceSet::~DeviceSet() {}

    void DeviceSet::AddDevice(Device* device) {
        devices_.push_back(device);
        for (const string& name :
                DeviceNameUtils::GetNamesForDeviceMappings(device->parsed_name())) {
            device_by_name_.insert({name, device});
        }
    }

    void DeviceSet::FindMatchingDevices(const DeviceNameUtils::ParsedName& spec,
            std::vector<Device*>* devices) const {
        // TODO(jeff): If we are going to repeatedly lookup the set of devices
        // for the same spec, maybe we should have a cache of some sort
        devices->clear();
        for (Device* d : devices_) {
            if (DeviceNameUtils::IsCompleteSpecification(spec, d->parsed_name())) {
                devices->push_back(d);
            }
        }
    }

    Device* DeviceSet::FindDeviceByName(const string& name) const {
        return gtl::FindPtrOrNull(device_by_name_, name);
    }

    // static
    int DeviceSet::DeviceTypeOrder(const DeviceType& d) {
        return DeviceFactory::DevicePriority(d.type_string());
    }

    static bool DeviceTypeComparator(const DeviceType& a, const DeviceType& b) {
        // First sort by prioritized device type (higher is preferred) and
        // then by device name (lexicographically).
        auto a_priority = DeviceSet::DeviceTypeOrder(a);
        auto b_priority = DeviceSet::DeviceTypeOrder(b);
        if (a_priority != b_priority) {
            return a_priority > b_priority;
        }

        return StringPiece(a.type()) < StringPiece(b.type());
    }

    std::vector<DeviceType> DeviceSet::PrioritizedDeviceTypeList() const {
        std::vector<DeviceType> result;
        std::set<string> seen;
        for (Device* d : devices_) {
            const auto& t = d->device_type();
            if (seen.insert(t).second) {
                result.emplace_back(t);
            }
        }
        std::sort(result.begin(), result.end(), DeviceTypeComparator);
        return result;
    }
}
