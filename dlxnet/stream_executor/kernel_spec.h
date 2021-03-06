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

// Kernel-loader specs are structures that describe how to load a data-parallel
// kernel on a given platform for subsequent launching. Headers that instantiate
// these data structures will typically be auto-generated. However, users can
// also instantiate them by hand.
//
// A kernel with the same exact functionality and type signature may be
// implemented on several different platforms. Typical usage is to create a
// singleton that describes how to load a kernel on the various supported
// platforms:
//
//  static const MultiKernelLoaderSpec &SaxpySpec() {
//    static auto *mkls =
//        (new MultiKernelLoaderSpec{4 /* = arity */})
//            ->AddCudaPtxOnDisk(ptx_file_path, ptx_kernelname)
//            ->AddOpenCLTextOnDisk(opencl_text_file_path, ocl_kernelname);
//    };
//
//    return *mkls;
//  }
//
// This lazily instantiates an object that describes how to load CUDA PTX
// present on disk that implements saxpy for the for the CUDA platform, or
// OpenCL text present on disk that implements saxpy for an OpenCL-based
// platform. The CudaPtxOnDisk and OpenCLTextOnDisk objects are subtypes of
// KernelLoaderSpec -- KernelLoaderSpec describes how to load a kernel for
// subsequent launching on a single platform.
//
// For the loader functionality that accepts these KernelLoaderSpecs in order
// to grab the kernel appropriately, see StreamExecutor::GetKernel().

#ifndef DLXNET_STREAM_EXECUTOR_KERNEL_SPEC_H_
#define DLXNET_STREAM_EXECUTOR_KERNEL_SPEC_H_

#include <stddef.h>

#include <map>
#include <memory>

#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "dlxnet/stream_executor/platform/logging.h"
#include "dlxnet/stream_executor/platform/port.h"

namespace stream_executor {
    // Describes how to load a kernel on a target platform.
    //
    // This is an abstract base class, subclassed for specific platforms.
    // The filename_or_text field represents the program location (i.e. PTX or
    // OpenCL loadable translation unit path) and is simply stored; whether it is a
    // filename or text is exposed via more specifically named accessors in
    // subclasses.
    //
    // These kernel loader specifications are typically auto-generated into header
    // files at build time, but can also be specified manually.
    class KernelLoaderSpec {
        public:
            virtual ~KernelLoaderSpec() {}

            // Returns the kernel name to load out of the program.
            const string &kernelname() const { return kernelname_; }

        protected:
            explicit KernelLoaderSpec(absl::string_view kernelname);

        private:
            // The kernel name that should be loaded out of the program description given
            // above.
            string kernelname_;

            SE_DISALLOW_COPY_AND_ASSIGN(KernelLoaderSpec);
    };

    // An abstract kernel loader spec that has an associated file path, where
    // there's a canonical suffix for the filename; e.g. see CudaPtxOnDisk whose
    // canonical filename suffix is ".ptx".
    class OnDiskKernelLoaderSpec : public KernelLoaderSpec {
        public:
            ~OnDiskKernelLoaderSpec() override {}

            // Returns the path to the on-disk loadable kernel file.
            const string &filename() const { return filename_; }

            // Returns the canonical suffix for this on-disk kernel loader spec format;
            // e.g. PTX files on disk have a canonical suffix of ".ptx".
            virtual const char *CanonicalSuffix() const = 0;

        protected:
            OnDiskKernelLoaderSpec(absl::string_view filename,
                    absl::string_view kernelname);

            string filename_;

        private:
            SE_DISALLOW_COPY_AND_ASSIGN(OnDiskKernelLoaderSpec);
    };

    // Kernel loader specification for OpenCL text that resides on disk.
    class OpenCLTextOnDisk : public OnDiskKernelLoaderSpec {
        public:
            OpenCLTextOnDisk(absl::string_view filename, absl::string_view kernelname);
            ~OpenCLTextOnDisk() override {}

            const char *CanonicalSuffix() const override { return ".ocl"; }

        private:
            SE_DISALLOW_COPY_AND_ASSIGN(OpenCLTextOnDisk);
    };

    // Kernel loader specification for OpenCL binary that resides on disk.
    class OpenCLBinaryOnDisk : public OnDiskKernelLoaderSpec {
        public:
            OpenCLBinaryOnDisk(absl::string_view filename, absl::string_view kernelname);
            ~OpenCLBinaryOnDisk() override {}

            const char *CanonicalSuffix() const override { return ".aocx"; }

        private:
            SE_DISALLOW_COPY_AND_ASSIGN(OpenCLBinaryOnDisk);
    };

    // Kernel loader specification for OpenCL text that resides in memory.
    class OpenCLTextInMemory : public KernelLoaderSpec {
        public:
            OpenCLTextInMemory(absl::string_view text, absl::string_view kernelname);
            ~OpenCLTextInMemory() override {}

            // Returns the OpenCL text contents.
            const string &text() const { return text_; }

        private:
            // OpenCL translation unit text contents in memory.
            string text_;

            SE_DISALLOW_COPY_AND_ASSIGN(OpenCLTextInMemory);
    };

    // Describes how to load a kernel on any subset of a number of target platforms.
    class MultiKernelLoaderSpec {
        public:
            explicit MultiKernelLoaderSpec(size_t arity);

            // Returns the number of arguments that this kernel accepts.
            size_t arity() const { return arity_; }

            // Convenience getters for testing whether these platform variants have
            // kernel loader specifications available.
            bool has_ocl_text_on_disk() const { return ocl_text_on_disk_ != nullptr; }
            bool has_ocl_binary_on_disk() const { return ocl_binary_on_disk_ != nullptr; }
            bool has_ocl_text_in_memory() const { return ocl_text_in_memory_ != nullptr; }

            // Accessors for platform variant kernel load specifications.
            // Precondition: corresponding has_* is true.
            const OpenCLTextOnDisk &ocl_text_on_disk() const {
                CHECK(has_ocl_text_on_disk());
                return *ocl_text_on_disk_;
            }
            const OpenCLBinaryOnDisk &ocl_binary_on_disk() const {
                CHECK(has_ocl_binary_on_disk());
                return *ocl_binary_on_disk_;
            }
            const OpenCLTextInMemory &ocl_text_in_memory() const {
                CHECK(has_ocl_text_in_memory());
                return *ocl_text_in_memory_;
            }


            // Builder-pattern-like methods for use in initializing a
            // MultiKernelLoaderSpec. Each of these should be used at most once for a
            // single MultiKernelLoaderSpec object. See file comment for example usage.
            //
            // Note that the kernelname parameter must be consistent with the kernel in
            // the PTX or OpenCL being loaded. Also be aware that in CUDA C++ the kernel
            // name may be mangled by the compiler if it is not declared in an
            // extern "C" scope.
            MultiKernelLoaderSpec *AddOpenCLTextOnDisk(absl::string_view filename,
                    absl::string_view kernelname);
            MultiKernelLoaderSpec *AddOpenCLBinaryOnDisk(absl::string_view filename,
                    absl::string_view kernelname);
            MultiKernelLoaderSpec *AddOpenCLTextInMemory(absl::string_view ocl_text,
                    absl::string_view kernelname);

        private:
            std::unique_ptr<OpenCLTextOnDisk>
                ocl_text_on_disk_;  // OpenCL text that resides on disk.
            std::unique_ptr<OpenCLBinaryOnDisk>
                ocl_binary_on_disk_;  // OpenCL binary that resides on disk.
            std::unique_ptr<OpenCLTextInMemory>
                ocl_text_in_memory_;  // OpenCL text that resides in memory.

            // Number of parameters that the kernel takes. (This is nicer to have in a
            // constexpr than having to determine it from the types via template
            // metaprogramming).
            size_t arity_;
    };
}//namespace stream_executor

#endif
