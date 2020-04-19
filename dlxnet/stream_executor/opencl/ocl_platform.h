#ifndef DLXNET_STREAM_EXECUTOR_OCL_OCL_PLATFORM_H_
#define DLXNET_STREAM_EXECUTOR_OCL_OCL_PLATFORM_H_

#include <memory>
#include "dlxnet/stream_executor/platform/port.h"
#include <vector>

#include "dlxnet/stream_executor/executor_cache.h"
#include "dlxnet/stream_executor/lib/statusor.h"
#include "dlxnet/stream_executor/multi_platform_manager.h"
#include "dlxnet/stream_executor/platform.h"
#include "dlxnet/stream_executor/platform/port.h"
#include "dlxnet/stream_executor/platform/thread_annotations.h"
#include "dlxnet/stream_executor/stream_executor_internal.h"
#include "dlxnet/stream_executor/stream_executor.h"
#include "dlxnet/stream_executor/trace_listener.h"

namespace stream_executor{
    // OpenCL-specific platform plugin, registered as a singleton value via module
    // initializer.
    class OCLPlatform : public Platform {
        public:
            OCLPlatform();
            ~OCLPlatform() override;

            // Platform interface implementation:
            // Returns the same value as kOCLPlatform above.
            Platform::Id id() const override;

            // Returns -1 as a sentinel on internal failure (and logs the error).
            int VisibleDeviceCount() const override;

            const string& Name() const override;

            port::StatusOr<std::unique_ptr<DeviceDescription>> DescriptionForDevice(
                    int ordinal) const override;

            port::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) override;

            port::StatusOr<StreamExecutor*> ExecutorForDeviceWithPluginConfig(
                    int ordinal, const PluginConfig& config) override;

            port::StatusOr<StreamExecutor*> GetExecutor(
                    const StreamExecutorConfig& config) override;

            port::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(
                    const StreamExecutorConfig& config) override;

            void RegisterTraceListener(std::unique_ptr<TraceListener> listener) override;

            void UnregisterTraceListener(TraceListener* listener) override;

        private:

            // This platform's name.
            string name_;

            // Cache of created executors.
            ExecutorCache executor_cache_;

            SE_DISALLOW_COPY_AND_ASSIGN(OCLPlatform);
    };
}//namespace stream_executor


#endif
