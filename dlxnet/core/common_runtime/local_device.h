#ifndef DLXNET_CORE_COMMON_RUNTIME_LOCAL_DEVICE_H_
#define DLXNET_CORE_COMMON_RUNTIME_LOCAL_DEVICE_H_
#include "dlxnet/core/public/session_options.h"
#include "dlxnet/core/common_runtime/device.h"
#include "dlxnet/core/framework/device_attributes.pb.h"

namespace dlxnet{
    // wrap eigen thread pool and its thread pool device
    class LocalDevice: public Device{
        public:
            LocalDevice(const SessionOptions& options,
                    const DeviceAttributes& attributes);
            ~LocalDevice() override;
        private:
            struct EigenThreadPoolInfo;
            std::unique_ptr<EigenThreadPoolInfo> owned_tp_info_;

            TF_DISALLOW_COPY_AND_ASSIGN(LocalDevice);
    };
}


#endif
