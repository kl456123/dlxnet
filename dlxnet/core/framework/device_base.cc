#define EIGEN_USE_THREADS
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "dlxnet/core/framework/device_base.h"
namespace dlxnet {
    DeviceBase::~DeviceBase() {
        for (auto& temp : eigen_cpu_devices_) {
            delete temp;
        }
        eigen_cpu_devices_.clear();
    }
    const DeviceAttributes& DeviceBase::attributes() const {
        LOG(FATAL) << "Device does not implement attributes()";
    }

    const string& DeviceBase::name() const {
        LOG(FATAL) << "Device does not implement name()";
    }
}
