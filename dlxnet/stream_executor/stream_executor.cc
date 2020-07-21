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

// Implements the StreamExecutor interface by passing through to its
// implementation_ value (in pointer-to-implementation style), which
// implements StreamExecutorInterface.

#include "dlxnet/stream_executor/stream_executor.h"

#include <atomic>
#include <memory>
#include <utility>

#include "absl/base/const_init.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/notification.h"
#include "dlxnet/core/util/env_var.h"
#include "dlxnet/stream_executor/lib/env.h"
#include "dlxnet/stream_executor/lib/error.h"
// #include "dlxnet/stream_executor/lib/stacktrace.h"
#include "dlxnet/stream_executor/lib/statusor.h"
#include "dlxnet/stream_executor/lib/threadpool.h"
#include "dlxnet/stream_executor/platform/port.h"
#include "dlxnet/stream_executor/stream_executor_internal.h"


namespace stream_executor{
    namespace {
        // Make sure the executor is done with its work; we know (because this isn't
        // publicly visible) that all enqueued work is quick.
        void BlockOnThreadExecutor(port::ThreadPool *executor) {
            absl::Notification n;
            executor->Schedule([&n]() { n.Notify(); });
            n.WaitForNotification();
        }

        // Get per-device memory limit in bytes. Returns 0 if
        // TF_PER_DEVICE_MEMORY_LIMIT_MB environment variable is not set.
        static int64 GetMemoryLimitBytes() {
            int64 value;
            SE_CHECK_OK(dlxnet::ReadInt64FromEnvVar("TF_PER_DEVICE_MEMORY_LIMIT_MB",
                        0, &value));
            return value * (1ll << 20);
        }
    }// namespace

    StreamExecutor::StreamExecutor(
            const Platform *platform,
            std::unique_ptr<internal::StreamExecutorInterface> implementation,
            int device_ordinal)
        : platform_(platform),
        implementation_(std::move(implementation)),
        device_ordinal_(device_ordinal),
        background_threads_(new port::ThreadPool(
                    port::Env::Default(), "stream_executor", kNumBackgroundThreads)),
        live_stream_count_(0),
        tracing_enabled_(false),
        mem_alloc_bytes_(0),
        memory_limit_bytes_(GetMemoryLimitBytes()),
        allocator_(this) {
            string name = absl::AsciiStrToLower(platform_->Name());
            if (name == "cuda") {
                platform_kind_ = PlatformKind::kCuda;
            } else if (name == "rocm") {
                platform_kind_ = PlatformKind::kROCm;
            } else if (name == "opencl") {
                platform_kind_ = PlatformKind::kOpenCL;
            } else if (name == "host") {
                platform_kind_ = PlatformKind::kHost;
            } else {
                platform_kind_ = PlatformKind::kInvalid;
            }
        }

    StreamExecutor::~StreamExecutor() {
        BlockOnThreadExecutor(background_threads_.get());

        if (live_stream_count_.load() != 0) {
            LOG(WARNING) << "Not all streams were deallocated at executor destruction "
                << "time. This may lead to unexpected/bad behavior - "
                << "especially if any stream is still active!";
        }

    }

    port::Status StreamExecutor::Init(DeviceOptions device_options) {
        return implementation_->Init(device_ordinal_, std::move(device_options));
    }

    port::Status StreamExecutor::Init() { return Init(DeviceOptions::Default()); }

    port::Status StreamExecutor::GetKernel(const MultiKernelLoaderSpec &spec,
            KernelBase *kernel) {
        return implementation_->GetKernel(spec, kernel);
    }

    void StreamExecutor::UnloadKernel(const KernelBase *kernel) {
        implementation_->UnloadKernel(kernel);
    }

    void StreamExecutor::Deallocate(DeviceMemoryBase *mem) {

        implementation_->Deallocate(mem);
        mem->Reset(nullptr, 0);
    }


    port::Status StreamExecutor::AllocateEvent(Event *event) {
        return implementation_->AllocateEvent(event);
    }

    port::Status StreamExecutor::DeallocateEvent(Event *event) {
        return implementation_->DeallocateEvent(event);
    }

    port::Status StreamExecutor::RecordEvent(Stream *stream, Event *event) {
        return implementation_->RecordEvent(stream, event);
    }

    port::Status StreamExecutor::WaitForEvent(Stream *stream, Event *event) {
        return implementation_->WaitForEvent(stream, event);
    }

    Event::Status StreamExecutor::PollForEventStatus(Event *event) {
        return implementation_->PollForEventStatus(event);
    }

    bool StreamExecutor::AllocateTimer(Timer *timer) {
        return implementation_->AllocateTimer(timer);
    }

    void StreamExecutor::DeallocateTimer(Timer *timer) {
        return implementation_->DeallocateTimer(timer);
    }

    bool StreamExecutor::StartTimer(Stream *stream, Timer *timer) {
        return implementation_->StartTimer(stream, timer);
    }

    bool StreamExecutor::StopTimer(Stream *stream, Timer *timer) {
        return implementation_->StopTimer(stream, timer);
    }

    bool StreamExecutor::AllocateStream(Stream *stream) {
        live_stream_count_.fetch_add(1, std::memory_order_relaxed);
        if (!implementation_->AllocateStream(stream)) {
            auto count = live_stream_count_.fetch_sub(1);
            CHECK_GE(count, 0) << "live stream count should not dip below zero";
            LOG(INFO) << "failed to allocate stream; live stream count: " << count;
            return false;
        }

        return true;
    }

    void StreamExecutor::DeallocateStream(Stream *stream) {
        implementation_->DeallocateStream(stream);
        CHECK_GE(live_stream_count_.fetch_sub(1), 0)
            << "live stream count should not dip below zero";
    }

    internal::StreamExecutorInterface *StreamExecutor::implementation() {
        return implementation_->GetUnderlyingExecutor();
    }


    StreamExecutorMemoryAllocator::StreamExecutorMemoryAllocator(
            StreamExecutor *executor)
        : DeviceMemoryAllocator(executor->platform()) {
            stream_executors_ = {executor};
        }

    StreamExecutorMemoryAllocator::StreamExecutorMemoryAllocator(
            const Platform *platform,
            absl::Span<StreamExecutor *const> stream_executors)
        : DeviceMemoryAllocator(platform),
        stream_executors_(stream_executors.begin(), stream_executors.end()) {}

    port::StatusOr<OwningDeviceMemory> StreamExecutorMemoryAllocator::Allocate(
            int device_ordinal, uint64 size, bool retry_on_failure,
            int64 memory_space) {
        TF_ASSIGN_OR_RETURN(StreamExecutor * executor,
                GetStreamExecutor(device_ordinal));
        DeviceMemoryBase result = executor->AllocateArray<uint8>(size, memory_space);
        if (size > 0 && result == nullptr) {
            return dlxnet::errors::ResourceExhausted(absl::StrFormat(
                        "Failed to allocate request for %s (%uB) on device ordinal %d",
                        dlxnet::strings::HumanReadableNumBytes(size), size,
                        device_ordinal));
        }
        VLOG(3) << absl::StreamFormat(
                "Allocated %s (%uB) on device ordinal %d: %p",
                dlxnet::strings::HumanReadableNumBytes(size), size, device_ordinal,
                result.opaque());
        return OwningDeviceMemory(result, device_ordinal, this);
    }

    port::Status StreamExecutorMemoryAllocator::Deallocate(int device_ordinal,
            DeviceMemoryBase mem) {
        if (!mem.is_null()) {
            TF_ASSIGN_OR_RETURN(StreamExecutor * executor,
                    GetStreamExecutor(device_ordinal));
            VLOG(3) << absl::StreamFormat("Freeing %p on device ordinal %d",
                    mem.opaque(), device_ordinal);
            executor->Deallocate(&mem);
        }
        return port::Status::OK();
    }

    port::StatusOr<StreamExecutor *>
        StreamExecutorMemoryAllocator::GetStreamExecutor(int device_ordinal) const {
            if (device_ordinal < 0) {
                return dlxnet::errors::InvalidArgument(absl::StrFormat(
                            "device ordinal value (%d) must be non-negative", device_ordinal));
            }
            for (StreamExecutor *se : stream_executors_) {
                if (se->device_ordinal() == device_ordinal) {
                    return se;
                }
            }
            return dlxnet::errors::NotFound(
                    absl::StrFormat("Device %s:%d present but not supported",
                        platform()->Name(), device_ordinal));
        }

    bool StreamExecutorMemoryAllocator::AllowsAsynchronousDeallocation() const {
        return false;
    }

    port::StatusOr<Stream *> StreamExecutorMemoryAllocator::GetStream(
            int device_ordinal) {
        CHECK(!AllowsAsynchronousDeallocation())
            << "The logic below only works for synchronous allocators";
        TF_ASSIGN_OR_RETURN(StreamExecutor * executor,
                GetStreamExecutor(device_ordinal));
        Stream *out = [&] {
            absl::MutexLock lock(&mutex_);
            if (!streams_.count(device_ordinal)) {
                auto p = streams_.emplace(std::piecewise_construct,
                        std::forward_as_tuple(device_ordinal),
                        std::forward_as_tuple(executor));
                p.first->second.Init();
                return &p.first->second;
            }
            return &streams_.at(device_ordinal);
        }();
        return out;
    }

    DeviceMemoryBase StreamExecutor::Allocate(uint64 size, int64 memory_space) {
        if (memory_limit_bytes_ > 0 &&
                mem_alloc_bytes_ + size > memory_limit_bytes_) {
            LOG(WARNING) << "Not enough memory to allocate " << size << " on device "
                << device_ordinal_
                << " within provided limit. [used=" << mem_alloc_bytes_
                << ", limit=" << memory_limit_bytes_ << "]";
            return DeviceMemoryBase();
        }
        DeviceMemoryBase buf = implementation_->Allocate(size, memory_space);
        return buf;
    }


    port::Status StreamExecutor::SynchronousMemcpyD2H(
            const DeviceMemoryBase &device_src, int64 size, void *host_dst) {

        port::Status result;

        result = implementation_->SynchronousMemcpy(host_dst, device_src, size);
        if (!result.ok()) {
            result = port::Status(
                    port::error::INTERNAL,
                    absl::StrFormat("failed to synchronously memcpy device-to-host: device "
                        "%p to host %p size %d: %s",
                        device_src.opaque(), host_dst, size,
                        result.ToString()));
        }

        return result;
    }

    port::Status StreamExecutor::SynchronousMemcpyH2D(
            const void *host_src, int64 size, DeviceMemoryBase *device_dst) {

        port::Status result;

        result = implementation_->SynchronousMemcpy(device_dst, host_src, size);
        if (!result.ok()) {
            result = port::Status(
                    port::error::INTERNAL,
                    absl::StrFormat("failed to synchronously memcpy host-to-device: host "
                        "%p to device %p size %d: %s",
                        host_src, device_dst->opaque(), size,
                        result.ToString()));
        }

        return result;
    }

    bool StreamExecutor::SynchronousMemcpy(DeviceMemoryBase *device_dst,
            const DeviceMemoryBase &device_src,
            uint64 size) {
        port::Status status = implementation_->SynchronousMemcpyDeviceToDevice(
                device_dst, device_src, size);
        if (!status.ok()) {
            LOG(ERROR) << "synchronous memcpy: " << status;
        }
        return status.ok();
    }

    bool StreamExecutor::Memcpy(Stream *stream, void *host_dst,
            const DeviceMemoryBase &device_src, uint64 size) {
        return implementation_->Memcpy(stream, host_dst, device_src, size);
    }

    bool StreamExecutor::Memcpy(Stream *stream, DeviceMemoryBase *device_dst,
            const void *host_src, uint64 size) {
        return implementation_->Memcpy(stream, device_dst, host_src, size);
    }

    bool StreamExecutor::MemcpyDeviceToDevice(Stream *stream,
            DeviceMemoryBase *device_dst,
            const DeviceMemoryBase &device_src,
            uint64 size) {
        return implementation_->MemcpyDeviceToDevice(stream, device_dst, device_src,
                size);
    }

    port::Status StreamExecutor::BlockHostUntilDone(Stream *stream) {
        port::Status result;
        result = implementation_->BlockHostUntilDone(stream);
        return result;
    }

    port::Status StreamExecutor::Launch(Stream *stream,
            const ThreadDim &thread_dims,
            const BlockDim &block_dims,
            const KernelBase &kernel,
            const KernelArgsArrayBase &args) {

        return implementation_->Launch(stream, thread_dims, block_dims, kernel, args);
    }

    bool StreamExecutor::DeviceMemoryUsage(int64 *free, int64 *total) const {
        return implementation_->DeviceMemoryUsage(free, total);
    }

    const DeviceDescription &StreamExecutor::GetDeviceDescription() const {
        if (device_description_ != nullptr) {
            return *device_description_;
        }

        device_description_ = CreateDeviceDescription();
        return *device_description_;
    }

    std::unique_ptr<DeviceDescription> StreamExecutor::CreateDeviceDescription()
        const {
            auto desc_status = implementation_->CreateDeviceDescription();
            return desc_status.ConsumeValueOrDie();
        }


}//namespace stream_executor
