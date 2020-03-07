/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef DLXNET_CORE_PLATFORM_STACKTRACE_H_
#define DLXNET_CORE_PLATFORM_STACKTRACE_H_

#include "dlxnet/core/platform/platform.h"

// Include appropriate platform-dependent implementation.
#if defined(PLATFORM_GOOGLE)
#include "dlxnet/core/platform/google/stacktrace.h"
#elif defined(PLATFORM_POSIX) || defined(PLATFORM_POSIX_ANDROID) ||    \
    defined(PLATFORM_GOOGLE_ANDROID) || defined(PLATFORM_POSIX_IOS) || \
    defined(PLATFORM_GOOGLE_IOS)
#include "dlxnet/core/platform/default/stacktrace.h"
#elif defined(PLATFORM_WINDOWS)
#include "dlxnet/core/platform/windows/stacktrace.h"
#else
#error Define the appropriate PLATFORM_<foo> macro for this platform
#endif

#endif  // DLXNET_CORE_PLATFORM_STACKTRACE_H_
