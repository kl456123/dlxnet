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

#include "dlxnet/stream_executor/stream.h"

#include "dlxnet/stream_executor/platform/port.h"

#include "absl/strings/str_cat.h"
#include "dlxnet/stream_executor/platform.h"
#include "dlxnet/stream_executor/platform/logging.h"
#include "dlxnet/stream_executor/stream_executor_internal.h"
#include "dlxnet/stream_executor/stream_executor.h"

namespace stream_executor{
    Stream::Stream(StreamExecutor *parent)
        : parent_(parent),
        implementation_(parent->implementation()->GetStreamImplementation()),
        allocated_(false),
        ok_(false){
            // VLOG_CALL(PARAM(parent));
        }

    Stream::Stream(StreamExecutor *parent,
            internal::StreamInterface *implementation)
        : parent_(parent),
        implementation_(implementation),
        allocated_(false),
        ok_(false){
            // VLOG_CALL(PARAM(parent), PARAM(implementation));
        }

    Stream::~Stream() {
        // VLOG_CALL();

        // Ensure the stream is completed.
        auto status = BlockHostUntilDone();
        if (!status.ok()) {
            LOG(WARNING) << "Error blocking host until done in stream destructor: "
                << status;
        }
        // temporary_memory_manager_.ForceDeallocateAll();
        // RunAfterBlockHostUntilDoneCallbacks();

        if (allocated_) {
            parent_->DeallocateStream(this);
        }
    }

    port::Status Stream::BlockHostUntilDone() {
        if (!ok()) {
            port::Status status = port::Status(
                    port::error::INTERNAL,
                    "stream did not block host until done; was already in an error state");
            // LOG(INFO) << DebugStreamPointers() << " " << status;
            return status;
        }

        port::Status error = parent_->BlockHostUntilDone(this);
        CheckError(error.ok());

        return error;
    }

    Stream &Stream::Init() {

        absl::MutexLock lock(&mu_);
        CHECK_EQ(false, allocated_)
            << "stream appears to already have been initialized";
        CHECK(!ok_) << "stream should be in !ok() state pre-initialization";

        if (parent_->AllocateStream(this)) {
            // Successful initialization!
            allocated_ = true;
            ok_ = true;
        } else {
            LOG(ERROR) << "failed to allocate stream during initialization";
        }

        return *this;
    }

    Stream &Stream::ThenMemcpy(void *host_dst, const DeviceMemoryBase &gpu_src,
            uint64 size) {
        // VLOG_CALL(PARAM(host_dst), PARAM(gpu_src), PARAM(size));

        if (ok()) {
            CheckError(parent_->Memcpy(this, host_dst, gpu_src, size));
        } else {
            LOG(INFO) << DebugStreamPointers()
                << " did not memcpy device-to-host; source: " << gpu_src.opaque();
        }
        return *this;
    }

    Stream &Stream::ThenMemcpy(DeviceMemoryBase *gpu_dst, const void *host_src,
            uint64 size) {
        // VLOG_CALL(PARAM(gpu_dst), PARAM(host_src), PARAM(size));

        if (ok()) {
            CheckError(parent_->Memcpy(this, gpu_dst, host_src, size));
        } else {
            LOG(INFO) << DebugStreamPointers()
                << " did not memcpy host-to-device; source: " << host_src;
        }
        return *this;
    }

    Stream &Stream::ThenMemcpy(DeviceMemoryBase *gpu_dst,
            const DeviceMemoryBase &gpu_src, uint64 size) {
        // VLOG_CALL(PARAM(gpu_dst), PARAM(gpu_src), PARAM(size));

        if (ok()) {
            CheckError(parent_->MemcpyDeviceToDevice(this, gpu_dst, gpu_src, size));
        } else {
            LOG(INFO) << DebugStreamPointers()
                << " did not memcpy gpu-to-gpu; source: " << &gpu_src;
        }
        return *this;
    }

    Stream &Stream::ThenWaitFor(Stream *other) {
        // VLOG_CALL(PARAM(other));

        CHECK(this != other) << "stream cannot wait for itself";
        if (ok() && other->ok()) {
            // CheckError(parent_->CreateStreamDependency(this, other));
        } else {
            // SetError();
            // LOG(INFO) << DebugStreamPointers() << " did not wait for "
            // << other->DebugStreamPointers();
        }
        return *this;
    }

    Stream &Stream::ThenWaitFor(Event *event) {
        // VLOG_CALL(PARAM(event));

        if (ok()) {
            port::Status status = parent_->WaitForEvent(this, event);
            if (!status.ok()) {
                LOG(ERROR) << "Error waiting for event in stream: "
                    << status.error_message()
                    << "; not marking stream as bad, as the Event object may be "
                    << "at fault. Monitor for further errors.";
            }
        } else {
            // LOG(INFO) << DebugStreamPointers() << " did not wait for an event.";
        }
        return *this;
    }

    string Stream::DebugStreamPointers() const {
        return absl::StrCat("[stream=", ",impl=", "]");
        // Relies on the ToVlogString(const void*) overload above.
        // return absl::StrCat("[stream=", ToVlogString(this),
        // ",impl=", ToVlogString(implementation_.get()), "]");
    }
}//namespace stream_executor
