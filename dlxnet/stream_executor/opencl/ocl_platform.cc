#include "dlxnet/stream_executor/opencl/ocl_platform.h"

#include "absl/base/const_init.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "dlxnet/stream_executor/lib/error.h"
#include "dlxnet/stream_executor/lib/initialize.h"
#include "dlxnet/stream_executor/lib/status.h"

#include "dlxnet/stream_executor/opencl/ocl_driver.h"
#include "dlxnet/stream_executor/opencl/ocl_gpu_executor.h"

namespace stream_executor{

    namespace {
        const Platform::Id kOCLPlatformId = 0;
        const DeviceOptions GetDeviceOptionsFromEnv() {
            return DeviceOptions::Default();
        }
    }// namespace

    OCLPlatform::OCLPlatform()
        : name_("OpenCL"){
            // init driver
            // just call it once
            OCLDriver::Init();
        }

    OCLPlatform::~OCLPlatform() {}

    Platform::Id OCLPlatform::id() const { return kOCLPlatformId; }

    int OCLPlatform::VisibleDeviceCount() const {
        if(!OCLDriver::Initialized()){
            return -1;
        }
        return OCLDriver::GetDeviceCount();
    }

    const string& OCLPlatform::Name() const { return name_; }

    port::StatusOr<StreamExecutor*> OCLPlatform::ExecutorForDevice(int ordinal) {
        StreamExecutorConfig config;
        config.ordinal = ordinal;
        config.plugin_config = PluginConfig();
        config.device_options = GetDeviceOptionsFromEnv();
        return GetExecutor(config);
    }

    port::StatusOr<StreamExecutor*> OCLPlatform::ExecutorForDeviceWithPluginConfig(
            int device_ordinal, const PluginConfig& plugin_config) {
        StreamExecutorConfig config;
        config.ordinal = device_ordinal;
        config.plugin_config = plugin_config;
        config.device_options = GetDeviceOptionsFromEnv();
        return GetExecutor(config);
    }

    port::StatusOr<StreamExecutor*> OCLPlatform::GetExecutor(
            const StreamExecutorConfig& config) {
        return executor_cache_.GetOrCreate(
                config, [&]() { return GetUncachedExecutor(config); });
    }

    port::StatusOr<std::unique_ptr<DeviceDescription>>
        OCLPlatform::DescriptionForDevice(int ordinal) const {
            return OCLExecutor::CreateDeviceDescription(ordinal);
        }

    port::StatusOr<std::unique_ptr<StreamExecutor>>
        OCLPlatform::GetUncachedExecutor(const StreamExecutorConfig& config) {
            auto executor = absl::make_unique<StreamExecutor>(
                    this, absl::make_unique<OCLExecutor>(config.plugin_config),
                    config.ordinal);
            auto init_status = executor->Init(config.device_options);
            if (!init_status.ok()) {
                return port::Status(
                        port::error::INTERNAL,
                        absl::StrFormat(
                            "failed initializing StreamExecutor for OpenCL device ordinal %d: %s",
                            config.ordinal, init_status.ToString()));
            }

            return std::move(executor);
        }

    void OCLPlatform::RegisterTraceListener(
            std::unique_ptr<TraceListener> listener) {
        LOG(FATAL) << "not yet implemented: register OpenCL trace listener";
    }

    void OCLPlatform::UnregisterTraceListener(TraceListener* listener) {
        LOG(FATAL) << "not yet implemented: unregister OpenCL trace listener";
    }

    static void InitializeOCLPlatform() {
        // Disabling leak checking, MultiPlatformManager does not destroy its
        // registered platforms.

        std::unique_ptr<OCLPlatform> platform(new OCLPlatform);
        SE_CHECK_OK(MultiPlatformManager::RegisterPlatform(std::move(platform)));
    }
} // namespace stream_executor

REGISTER_MODULE_INITIALIZER(ocl_platform,
        stream_executor::InitializeOCLPlatform());
