#ifndef DLXNET_STREAM_EXECUTOR_STREAM_EXECUTOR_INTERNAL_H_
#define DLXNET_STREAM_EXECUTOR_STREAM_EXECUTOR_INTERNAL_H_
#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "absl/types/optional.h"

#include "dlxnet/stream_executor/allocator_stats.h"
#include "dlxnet/stream_executor/lib/status.h"
#include "dlxnet/stream_executor/lib/statusor.h"
#include "dlxnet/stream_executor/device_options.h"
#include "dlxnet/stream_executor/device_description.h"
#include "dlxnet/stream_executor/kernel_cache_config.h"
#include "dlxnet/stream_executor/stream.h"
#include "dlxnet/stream_executor/launch_dim.h"
#include "dlnxet/stream_executor/device_memory.h"
#include "dlxnet/stream_executor/event.h"
#include "dlxnet/stream_executor/timer.h"
#include "dlxnet/stream_executor/trace_listener.h"

namespace stream_executor{
    class Stream;
    class Timer;

    namespace internal{
        // Platform-dependent interface class for the generic Events interface, in
        // the PIMPL style.
        class EventInterface {
            public:
                EventInterface() {}
                virtual ~EventInterface() {}

            private:
                SE_DISALLOW_COPY_AND_ASSIGN(EventInterface);
        };

        // Pointer-to-implementation object type (i.e. the KernelBase class delegates to
        // this interface) with virtual destruction. This class exists for the
        // platform-dependent code to hang any kernel data/resource info/functionality
        // off of.
        class KernelInterface {
            public:
                // Default constructor for the abstract interface.
                KernelInterface() {}

                // Default destructor for the abstract interface.
                virtual ~KernelInterface() {}

                // Returns the number of formal parameters that this kernel accepts.
                virtual unsigned Arity() const = 0;

                // Sets the preferred cache configuration.
                virtual void SetPreferredCacheConfig(KernelCacheConfig config) = 0;

                // Gets the preferred cache configuration.
                virtual KernelCacheConfig GetPreferredCacheConfig() const = 0;

            private:
                SE_DISALLOW_COPY_AND_ASSIGN(KernelInterface);
        };

        // Pointer-to-implementation object type (i.e. the Stream class delegates to
        // this interface) with virtual destruction. This class exists for the
        // platform-dependent code to hang any kernel data/resource info/functionality
        // off of.
        class StreamInterface {
            public:
                // Default constructor for the abstract interface.
                StreamInterface() {}

                // Default destructor for the abstract interface.
                virtual ~StreamInterface() {}

            private:
                SE_DISALLOW_COPY_AND_ASSIGN(StreamInterface);
        };

        // Pointer-to-implementation object type (i.e. the Timer class delegates to
        // this interface) with virtual destruction. This class exists for the
        // platform-dependent code to hang any timer data/resource info/functionality
        // off of.
        class TimerInterface {
            public:
                // Default constructor for the abstract interface.
                TimerInterface() {}

                // Default destructor for the abstract interface.
                virtual ~TimerInterface() {}

                // Returns the number of microseconds elapsed in a completed timer.
                virtual uint64 Microseconds() const = 0;

                // Returns the number of nanoseconds elapsed in a completed timer.
                virtual uint64 Nanoseconds() const = 0;

            private:
                SE_DISALLOW_COPY_AND_ASSIGN(TimerInterface);
        };

        // Interface for the different StreamExecutor platforms (i.e. CUDA, OpenCL).
        //
        // Various platforms will provide an implementation that satisfy this interface.
        class StreamExecutorInterface {
            public:
                // Default constructor for the abstract interface.
                StreamExecutorInterface() {}

                // Default destructor for the abstract interface.
                virtual ~StreamExecutorInterface() {}

                // Returns the (transitively) wrapped executor if this executor is
                // wrapping another executor; otherwise, returns this.
                virtual StreamExecutorInterface *GetUnderlyingExecutor() { return this; }

                // See the StreamExecutor interface for comments on the same-named methods.
                virtual port::Status Init(int device_ordinal,
                        DeviceOptions device_options) = 0;

                virtual port::Status Launch(Stream *stream, const ThreadDim &thread_dims,
                        const BlockDim &block_dims, const KernelBase &k,
                        const KernelArgsArrayBase &args) {
                    return port::UnimplementedError("Not Implemented");
                }
                virtual DeviceMemoryBase Allocate(uint64 size, int64 memory_space) = 0;
                DeviceMemoryBase Allocate(uint64 size) {
                    return Allocate(size, /*memory_space=*/0);
                }
                virtual void Deallocate(DeviceMemoryBase *mem) = 0;

                virtual bool AllocateStream(Stream *stream) = 0;
                virtual void DeallocateStream(Stream *stream) = 0;
                virtual bool AllocateTimer(Timer *timer) = 0;
                virtual void DeallocateTimer(Timer *timer) = 0;
                virtual bool StartTimer(Stream *stream, Timer *timer) = 0;
                virtual bool StopTimer(Stream *stream, Timer *timer) = 0;
                virtual port::Status BlockHostUntilDone(Stream *stream) = 0;
                virtual port::Status GetStatus(Stream *stream) {
                    return port::Status(port::error::UNIMPLEMENTED,
                            "GetStatus is not supported on this executor.");
                }
                virtual int PlatformDeviceCount() = 0;
                virtual bool DeviceMemoryUsage(int64 *free, int64 *total) const {
                    return false;
                }

                // Creates a new DeviceDescription object. Ownership is transferred to the
                // caller.
                virtual port::StatusOr<std::unique_ptr<DeviceDescription>>
                    CreateDeviceDescription() const = 0;

                // Attempts to register the provided TraceListener with the device-specific
                // Executor implementation. When this is called, the PIMPL interface has
                // already taken ownership of the object and is managing the generic tracing
                // events. The device-specific implementation must determine if the passed
                // listener is of a type appropriate for it to trace during registration (and
                // before dispatching events to it).
                // Returns true if the listener was successfully registered, false otherwise.
                // Does not take ownership of listener.
                virtual bool RegisterTraceListener(TraceListener *listener) { return false; }

                // Unregisters the specified listener from the device-specific Executor.
                // Returns true if the listener was successfully registered, false otherwise.
                virtual bool UnregisterTraceListener(TraceListener *listener) {
                    return false;
                }

                // Each call creates a new instance of the platform-specific implementation of
                // the corresponding interface type.
                virtual std::unique_ptr<EventInterface> CreateEventImplementation() = 0;
                virtual std::unique_ptr<KernelInterface> CreateKernelImplementation() = 0;
                virtual std::unique_ptr<StreamInterface> GetStreamImplementation() = 0;
                virtual std::unique_ptr<TimerInterface> GetTimerImplementation() = 0;

                // Return allocator statistics.
                virtual absl::optional<AllocatorStats> GetAllocatorStats() {
                    return absl::nullopt;
                }
            private:
                SE_DISALLOW_COPY_AND_ASSIGN(StreamExecutorInterface);
        };

    }//namespace internal

}//namespace stream_executor


#endif
