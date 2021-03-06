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

#include "dlxnet/stream_executor/kernel_spec.h"
#include "absl/strings/string_view.h"

namespace stream_executor {
    KernelLoaderSpec::KernelLoaderSpec(absl::string_view kernelname)
        : kernelname_(string(kernelname)) {}

    OnDiskKernelLoaderSpec::OnDiskKernelLoaderSpec(absl::string_view filename,
            absl::string_view kernelname)
        : KernelLoaderSpec(kernelname), filename_(string(filename)) {}

    MultiKernelLoaderSpec::MultiKernelLoaderSpec(size_t arity) : arity_(arity) {}

    OpenCLTextOnDisk::OpenCLTextOnDisk(absl::string_view filename,
            absl::string_view kernelname)
        : OnDiskKernelLoaderSpec(filename, kernelname) {}

    OpenCLTextInMemory::OpenCLTextInMemory(absl::string_view text,
            absl::string_view kernelname)
        : KernelLoaderSpec(kernelname), text_(text) {}

    OpenCLBinaryOnDisk::OpenCLBinaryOnDisk(absl::string_view filename,
            absl::string_view kernelname)
        : OnDiskKernelLoaderSpec(filename, kernelname) {}

    MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddOpenCLTextOnDisk(
            absl::string_view filename, absl::string_view kernelname) {
        CHECK(ocl_text_on_disk_ == nullptr);
        ocl_text_on_disk_.reset(new OpenCLTextOnDisk{filename, kernelname});
        return this;
    }

    MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddOpenCLBinaryOnDisk(
            absl::string_view filename, absl::string_view kernelname) {
        CHECK(ocl_binary_on_disk_ == nullptr);
        ocl_binary_on_disk_.reset(new OpenCLBinaryOnDisk{filename, kernelname});
        return this;
    }

    MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddOpenCLTextInMemory(
            absl::string_view filename, absl::string_view kernelname) {
        CHECK(ocl_text_in_memory_ == nullptr);
        ocl_text_in_memory_.reset(new OpenCLTextInMemory{filename, kernelname});
        return this;
    }
}//namespace stream_executor
