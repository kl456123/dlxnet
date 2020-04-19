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
    };

}//namespace stream_executor
