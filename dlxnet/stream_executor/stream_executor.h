/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef DLXNET_STREAM_EXECUTOR_STREAM_EXECUTOR_PIMPL_H_
#define DLXNET_STREAM_EXECUTOR_STREAM_EXECUTOR_PIMPL_H_

#include <atomic>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

#include "absl/base/macros.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "dlxnet/stream_executor/device_memory_allocator.h"
#include "dlxnet/stream_executor/lib/status.h"
#include "dlxnet/stream_executor/lib/statusor.h"
#include "dlxnet/stream_executor/lib/threadpool.h"
#include "dlxnet/stream_executor/platform.h"
#include "dlxnet/stream_executor/platform/logging.h"
#include "dlxnet/stream_executor/platform/port.h"
#include "dlxnet/stream_executor/platform/thread_annotations.h"
// #include "dlxnet/stream_executor/rng.h"
#include "dlxnet/stream_executor/shared_memory_config.h"
#include "dlxnet/stream_executor/stream.h"
#include "dlxnet/stream_executor/stream_executor_internal.h"
#include "dlxnet/stream_executor/trace_listener.h"

namespace stream_executor{
    // A StreamExecutor manages a single device, in terms of executing work (kernel
    // launches) and memory management (allocation/deallocation, memory copies to
    // and from the device). It is conceptually the "handle" for a device -- Stream
    // objects, which are used to enqueue work to run on the
    // coprocessor have a StreamExecutor instance as their "parent" object.
    //
    // StreamExecutor objects have an underlying platform that is specified up
    // front;
    // e.g. either it is a CUDA or OpenCL executor.
    //
    // Thread-safe after initialization.
    // StreamExecutor interface should not be invoked from a signal handler.
    class StreamExecutor {
        public:
            StreamExecutor(
                    const Platform *platform,
                    std::unique_ptr<internal::StreamExecutorInterface> implementation,
                    int device_ordinal);

            ~StreamExecutor();

            port::Status Init();
            port::Status Init(DeviceOptions device_options);

            // Returns a reference to the platform that created this executor.
            const Platform *platform() const { return platform_; }

            // Retrieves (loads) a kernel for the platform this StreamExecutor is acting
            // upon, if one exists.
            //
            // Parameters:
            //   spec: The MultiKernelLoaderSpec is usually generated as a compile-time
            //    constant into an appropriate namespace. For example, see
            //    stream_executor::executor_sample::kKernelLoaderSpecs, from which a
            //    MultiKernelLoaderSpec is selected.
            //   kernel: Outparam that the kernel is loaded into. A given Kernel
            //    instantiation should not be loaded into more than once.
            //
            // If an error occurs, or there is no kernel available for the StreamExecutor
            // platform, error status is returned.
            port::Status GetKernel(const MultiKernelLoaderSpec &spec, KernelBase *kernel);

            // Releases any state associated with the previously loaded kernel.
            void UnloadKernel(const KernelBase *kernel);

            // Synchronously allocates an array on the device of type T with element_count
            // elements.
            template <typename T>
                DeviceMemory<T> AllocateArray(uint64 element_count, int64 memory_space = 0);

            // As AllocateArray(), but returns a ScopedDeviceMemory<T>.
            template <typename T>
                ScopedDeviceMemory<T> AllocateOwnedArray(uint64 element_count) {
                    return ScopedDeviceMemory<T>(this, AllocateArray<T>(element_count));
                }

            // Convenience wrapper that allocates space for a single element of type T in
            // device memory.
            template <typename T>
                DeviceMemory<T> AllocateScalar() {
                    return AllocateArray<T>(1);
                }

            // As AllocateScalar(), but returns a ScopedDeviceMemory<T>.
            template <typename T>
                ScopedDeviceMemory<T> AllocateOwnedScalar() {
                    return AllocateOwnedArray<T>(1);
                }

            // Synchronously allocates a scalar of type T on the device that is (POD)
            // zero-byte initialized.
            template <typename T>
                DeviceMemory<T> AllocateZeroed();

            // As AllocateZeroed(), but returns a ScopedDeviceMemory<T>.
            template <typename T>
                ScopedDeviceMemory<T> AllocateOwnedZeroed() {
                    return ScopedDeviceMemory<T>(this, AllocateZeroed<T>());
                }

            // Deallocate the DeviceMemory previously allocated via this interface.
            // Deallocation of a nullptr-representative value is permitted.
            //
            // Resets the internal contents of mem to be null-representative, but this
            // null-out effect should not be relied upon in client code.
            void Deallocate(DeviceMemoryBase *mem);

            // Returns the device ordinal that this StreamExecutor was initialized with.
            // Meaningless before initialization.
            int device_ordinal() const { return device_ordinal_; }

            // Returns a borrowed pointer to the underlying StreamExecutor implementation.
            internal::StreamExecutorInterface *implementation();

            // Registers a trace listener to receive callbacks for only a single
            // StreamExecutor instance.
            // To register a listener for all executors for a given platform, see
            // Platform::RegisterTraceListener().
            // Does not take ownership of listener.
            void RegisterTraceListener(TraceListener* listener);

            // Removes a TraceListener from this StreamExecutor instance.
            // Returns false (and logs) in cases where the argument listener was not
            // previously registered.
            bool UnregisterTraceListener(TraceListener* listener);

            // Return allocator statistics.
            absl::optional<AllocatorStats> GetAllocatorStats();

            // Return an allocator which delegates to this stream executor for memory
            // allocation.
            StreamExecutorMemoryAllocator *GetAllocator() { return &allocator_; }

            // Allocates timer resources on the underlying platform and initializes its
            // internals.
            bool AllocateTimer(Timer *timer);

            // Deallocates timer resources on the underlying platform.
            void DeallocateTimer(Timer *timer);

            // Records a start event for an interval timer.
            bool StartTimer(Stream *stream, Timer *timer);

            // Records a stop event for an interval timer.
            bool StopTimer(Stream *stream, Timer *timer);

            // Performs platform-specific allocation and initialization of an event.
            port::Status AllocateEvent(Event *event);

            // Performs platform-specific deallocation and cleanup of an event.
            port::Status DeallocateEvent(Event *event);

            // Inserts the specified event at the end of the specified stream.
            port::Status RecordEvent(Stream *stream, Event *event);

            // Wait for the specified event at the end of the specified stream.
            port::Status WaitForEvent(Stream *stream, Event *event);

            // Requests the current status of the event from the underlying platform.
            Event::Status PollForEventStatus(Event *event);

            // Allocates stream resources on the underlying platform and initializes its
            // internals.
            bool AllocateStream(Stream *stream);

            // Deallocates stream resources on the underlying platform.
            void DeallocateStream(Stream *stream);

            // Same as SynchronousMemcpy(DeviceMemoryBase*, ...) above.
            port::Status SynchronousMemcpyH2D(const void *host_src, int64 size,
                    DeviceMemoryBase *device_dst);

            // Alternative interface for memcpying from host to device that takes an
            // array slice. Checks that the destination size can accommodate the host
            // slice size.
            template <class T>
                port::Status SynchronousMemcpyH2D(port::ArraySlice<T> host_src,
                        DeviceMemoryBase *device_dst) {
                    auto host_size = host_src.size() * sizeof(T);
                    CHECK(device_dst->size() == 0 || device_dst->size() >= host_size);
                    return SynchronousMemcpyH2D(host_src.begin(), host_size, device_dst);
                }

            // Same as SynchronousMemcpy(void*, ...) above.
            port::Status SynchronousMemcpyD2H(const DeviceMemoryBase &device_src,
                    int64 size, void *host_dst);

            // Alternative interface for memcpying from device to host that takes an
            // array slice. Checks that the destination size can accommodate the host
            // slice size.
            template <typename T>
                port::Status SynchronousMemcpyD2H(const DeviceMemory<T> &device_src,
                        port::MutableArraySlice<T> host_dst) {
                    auto host_size = host_dst.size() * sizeof(T);
                    CHECK(device_src.size() == 0 || host_size >= device_src.size());
                    return SynchronousMemcpyD2H(device_src, host_size, host_dst.begin());
                }

            // Blocks the caller while a data segment of the given size is copied from the
            // device source to the device destination.
            bool SynchronousMemcpy(DeviceMemoryBase *device_dst,
                    const DeviceMemoryBase &device_src,
                    uint64 size) SE_MUST_USE_RESULT;

            // Warning: use Stream::ThenLaunch instead, this method is not for general
            // consumption. However, this is the only way to launch a kernel for which
            // the type signature is only known at runtime; say, if an application
            // supports loading/launching kernels with arbitrary type signatures.
            // In this case, the application is expected to know how to do parameter
            // packing that obeys the contract of the underlying platform implementation.
            //
            // Launches a data parallel kernel with the given thread/block
            // dimensionality and already-packed args/sizes to pass to the underlying
            // platform driver.
            //
            // This is called by Stream::Launch() to delegate to the platform's launch
            // implementation in StreamExecutorInterface::Launch().
            port::Status Launch(Stream *stream, const ThreadDim &thread_dims,
                    const BlockDim &block_dims, const KernelBase &kernel,
                    const KernelArgsArrayBase &args);

        private:
            friend class Event;
            friend class Stream;
            friend class Timer;
            template <typename... Params>
                friend class TypedKernel;
            // Synchronously allocates size bytes on the underlying platform and returns
            // a DeviceMemoryBase representing that allocation. In the case of failure,
            // nullptr is returned.
            DeviceMemoryBase Allocate(uint64 size, int64 memory_space);

            // Causes the host code to synchronously wait for operations entrained onto
            // stream to complete. Effectively a join on the asynchronous device
            // operations enqueued on the stream before this program point.
            port::Status BlockHostUntilDone(Stream *stream);

            // Without blocking the device, retrieve the current stream status.
            port::Status GetStatus(Stream *stream);

            // Reader/writer lock for class-static StreamExecutor members.
            static absl::Mutex static_mu_;

            // Reader/writer lock for mutable data structures on this StreamExecutor.
            //
            // Mutable so that caching functions (like DeviceDescription, AsBlas, etc.)
            // can acquire the lock on their first (mutating) call as well.
            mutable absl::Mutex mu_;

            // Reference to the platform that created this executor.
            const Platform *platform_;

            // Pointer to the platform-specific-interface implementation. This is
            // delegated to by the interface routines in pointer-to-implementation
            // fashion.
            std::unique_ptr<internal::StreamExecutorInterface> implementation_;

            // Slot to cache the owned DeviceDescription for the underlying device
            // once it has been quieried from DeviceDescription().
            mutable std::unique_ptr<DeviceDescription> device_description_
                GUARDED_BY(mu_);

            // The kind of the underlying platform that is being targeted, as passed
            // during construction.
            //
            // Immutable post-initialization.
            PlatformKind platform_kind_;

            // The device ordinal that this object was initialized with.
            //
            // Immutable post-initialization.
            int device_ordinal_;

            // Executor for handling host callback work that cannot be performed
            // by a host callback thread - for example, cleanup after a host BLAS routine
            // (which may make device API calls). This work cannot block the host
            // callback thread, will be completed asynchronously, and should be treated
            // as fire-and-forget. Assume no ordering guarantees WRT the tasks enqueued
            // here.
            //
            // Immutable post-initialization. Object is thread-safe.
            std::unique_ptr<port::ThreadPool> background_threads_;

            // Counter for the current number of live streams. This is used to check
            // for accidentally-outstanding streams at StreamExecutor teardown time, as
            // well
            // as to indicate leaks (via a large outstanding count being logged) in the
            // case we can't allocate more streams.
            std::atomic_int_fast32_t live_stream_count_;

            // Only one worker thread is needed; little work will be done by the
            // executor.
            static const int kNumBackgroundThreads = 1;

            // Indicates if StreamExecutor operation tracing should be performed.
            bool tracing_enabled_;

            // The set of TraceListeners registered for this StreamExecutor.
            std::set<TraceListener*> listeners_ GUARDED_BY(mu_);

            // Allocated memory in bytes.
            int64 mem_alloc_bytes_;

            // Memory limit in bytes. Value less or equal to 0 indicates there is no
            // limit.
            int64 memory_limit_bytes_;

            StreamExecutorMemoryAllocator allocator_;

            SE_DISALLOW_COPY_AND_ASSIGN(StreamExecutor);

    };

    // inline
    template <typename T>
        inline DeviceMemory<T> StreamExecutor::AllocateArray(uint64 element_count,
                int64 memory_space) {
            uint64 bytes = sizeof(T) * element_count;
            return DeviceMemory<T>(Allocate(bytes, memory_space));
        }

    template <typename... Params, typename... Args>
        inline Stream &Stream::ThenLaunch(ThreadDim thread_dims, BlockDim block_dims,
                const TypedKernel<Params...> &kernel,
                Args... args) {
            KernelInvocationChecker<std::tuple<Params...>,
            std::tuple<Args...>>::CheckAllStaticAssert();
            if (ok()) {
                // This is the core that allows type-safe kernel launching.
                // Since the platforms take kernel arguments as tuples of (void *, size),
                // we pack the variadic parameters passed as ...args into the desired
                // tuple form and pass that packed form to the StreamExecutor::Launch()
                // implementation.
                KernelArgsArray<sizeof...(args)> kernel_args;
                kernel.PackParams(&kernel_args, args...);
                DCHECK(parent_ != nullptr);
                bool ok =
                    parent_->Launch(this, thread_dims, block_dims, kernel, kernel_args)
                    .ok();
                if (!ok) {
                    SetError();
                    LOG(WARNING) << "parent failed to launch kernel: " << &kernel;
                }
            }
            return *this;
        }

    template <typename ElemT>
        ScopedDeviceMemory<ElemT>::ScopedDeviceMemory(StreamExecutor *parent,
                DeviceMemoryBase value)
        : wrapped_(value),
        device_ordinal_(parent->device_ordinal()),
        allocator_(parent->GetAllocator()) {}

    template <typename ElemT>
        ScopedDeviceMemory<ElemT>::ScopedDeviceMemory(
                StreamExecutor *parent, std::initializer_list<ElemT> values)
        : ScopedDeviceMemory(parent, parent->AllocateArray<ElemT>(values.size())) {
            if (ptr() != nullptr) {
                std::vector<ElemT> local(values);
                if (!parent->SynchronousMemcpy(ptr(), const_cast<const ElemT *>(&local[0]),
                            ptr()->size())) {
                    TF_CHECK_OK(Free());
                }
            }
        }

}//namespace stream_executor

#endif
