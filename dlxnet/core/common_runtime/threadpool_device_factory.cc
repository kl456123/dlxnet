#include <vector>

#include "absl/memory/memory.h"

#include "dlxnet/core/common_runtime/threadpool_device.h"
#include "dlxnet/core/common_runtime/device_factory.h"
#include "dlxnet/core/framework/allocator.h"
#include "dlxnet/core/platform/numa.h"



namespace dlxnet{
    class ThreadPoolDeviceFactory : public DeviceFactory{
        public:
            Status ListPhysicalDevices(std::vector<string>* devices) override {
                devices->push_back("/physical_device:CPU:0");

                return Status::OK();
            }
            Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                    std::vector<std::unique_ptr<Device>>* devices)override{
                // how many device used for this type
                int n = 1;
                auto iter = options.config.device_count().find("CPU");
                if (iter != options.config.device_count().end()) {
                    n = iter->second;
                }
                for (int i = 0; i < n; i++){
                    string name = strings::StrCat(name_prefix, "/device:CPU:", i);
                    std::unique_ptr<ThreadPoolDevice> tpd;
                    tpd = absl::make_unique<ThreadPoolDevice>(options, name, Bytes(256 << 20),
                            DeviceLocality(), cpu_allocator(port::kNUMANoAffinity));
                    devices->push_back(std::move(tpd));
                }
                return Status::OK();
            }

    };

    REGISTER_LOCAL_DEVICE_FACTORY("CPU", ThreadPoolDeviceFactory, 60);

}// namespace dlxnet
