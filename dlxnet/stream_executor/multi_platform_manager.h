#ifndef DLXNET_STREAM_EXECUTOR_MULTI_PLATFORM_MANAGER_H_
#define DLXNET_STREAM_EXECUTOR_MULTI_PLATFORM_MANAGER_H_
#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "dlxnet/stream_executor/platform.h"
#include "dlxnet/stream_executor/platform/port.h"
#include "dlxnet/stream_executor/lib/status.h"
#include "dlxnet/stream_executor/lib/statusor.h"

namespace stream_executor{
    // Manages multiple platforms that may be present on the current machine.
    class MultiPlatformManager{
        public:
            // Registers a platform object, returns an error status if the platform is
            // already registered. The associated listener, if not null, will be used to
            // trace events for ALL executors for that platform.
            // Takes ownership of platform.
            static port::Status RegisterPlatform(std::unique_ptr<Platform> platform);

            // Retrieves the platform registered with the given platform name (e.g.
            // "CUDA", "OpenCL", ...) or id (an opaque, comparable value provided by the
            // Platform's Id() method).
            //
            // If the platform has not already been initialized, it will be initialized
            // with a default set of parameters.
            //
            // If the requested platform is not registered, an error status is returned.
            // Ownership of the platform is NOT transferred to the caller --
            // the MultiPlatformManager owns the platforms in a singleton-like fashion.
            static port::StatusOr<Platform*> PlatformWithName(absl::string_view target);
            static port::StatusOr<Platform*> PlatformWithId(const Platform::Id& id);

            // Retrives the platforms satisfying the given filter, i.e. returns true.
            // Returned Platforms are always initialized.
            static port::StatusOr<std::vector<Platform*>> PlatformsWithFilter(
                    const std::function<bool(const Platform*)>& filter);

            static std::vector<Platform*> AllPlatforms();

            // Although the MultiPlatformManager "owns" its platforms, it holds them as
            // undecorated pointers to prevent races during program exit (between this
            // object's data and the underlying platforms (e.g., CUDA, OpenCL).
            // Because certain platforms have unpredictable deinitialization
            // times/sequences, it is not possible to strucure a safe deinitialization
            // sequence. Thus, we intentionally "leak" allocated platforms to defer
            // cleanup to the OS. This should be acceptable, as these are one-time
            // allocations per program invocation.
            // The MultiPlatformManager should be considered the owner
            // of any platforms registered with it, and leak checking should be disabled
            // during allocation of such Platforms, to avoid spurious reporting at program
            // exit.

            // Interface for a listener that gets notfied at certain events.
            class Listener {
                public:
                    virtual ~Listener() = default;
                    // Callback that is invoked when a Platform is registered.
                    virtual void PlatformRegistered(Platform* platform) = 0;
            };
            // Registers a listeners to receive notifications about certain events.
            // Precondition: No Platform has been registered yet.
            static port::Status RegisterListener(std::unique_ptr<Listener> listener);
    };
}// namespace stream_executor


#endif
