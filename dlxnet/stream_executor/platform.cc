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

#include "dlxnet/stream_executor/platform.h"

#include "dlxnet/stream_executor/platform/port.h"

#include "absl/strings/str_cat.h"
#include "dlxnet/stream_executor/lib/error.h"
#include "dlxnet/core/platform/logging.h"
// #include "dlxnet/stream_executor/stream_executor_pimpl.h"

namespace stream_executor {
    string PlatformKindString(PlatformKind kind) {
        switch (kind) {
            case PlatformKind::kCuda:
                return "CUDA";
            case PlatformKind::kROCm:
                return "ROCm";
            case PlatformKind::kOpenCL:
                return "OpenCL";
            case PlatformKind::kHost:
                return "Host";
            case PlatformKind::kMock:
                return "Mock";
            default:
                return absl::StrCat("InvalidPlatformKind(", static_cast<int>(kind), ")");
        }
    }

    PlatformKind PlatformKindFromString(string kind) {
        for (int i = 0; i < static_cast<int>(PlatformKind::kSize); ++i) {
            if (kind == PlatformKindString(static_cast<PlatformKind>(i))) {
                return static_cast<PlatformKind>(i);
            }
        }

        return PlatformKind::kInvalid;
    }

    bool PlatformIsRunnable(PlatformKind kind) {
        switch (kind) {
            case PlatformKind::kCuda:
            case PlatformKind::kROCm:
            case PlatformKind::kOpenCL:
            case PlatformKind::kHost:
                return true;
            default:
                return false;
        }
    }

    bool PlatformIsRunnableOnDevice(PlatformKind kind) {
        switch (kind) {
            case PlatformKind::kCuda:
            case PlatformKind::kROCm:
            case PlatformKind::kOpenCL:
                return true;
            default:
                return false;
        }
    }

    void CheckPlatformKindIsValid(PlatformKind kind) {
        CHECK(static_cast<int>(PlatformKind::kCuda) <= static_cast<int>(kind) &&
                static_cast<int>(kind) <= static_cast<int>(PlatformKind::kMock))
            << "invalid GPU executor kind: " << PlatformKindString(kind);
    }

    StreamExecutorConfig::StreamExecutorConfig()
        : ordinal(-1), device_options(DeviceOptions::Default()) {}

    StreamExecutorConfig::StreamExecutorConfig(int ordinal_in)
        : ordinal(ordinal_in), device_options(DeviceOptions::Default()) {}

    Platform::~Platform() {}

    bool Platform::Initialized() const { return true; }

    port::Status Platform::Initialize(
            const std::map<string, string> &platform_options) {
        if (!platform_options.empty()) {
            return port::Status(port::error::UNIMPLEMENTED,
                    "this platform does not support custom initialization");
        }
        return port::Status::OK();
    }

    port::Status Platform::ForceExecutorShutdown() {
        return port::Status(port::error::UNIMPLEMENTED,
                "executor shutdown is not supported on this platform");
    }
}//namespace stream_executor
