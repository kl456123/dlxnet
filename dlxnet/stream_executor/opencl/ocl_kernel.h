/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// The CUDA implementation of the StreamExecutorInterface functionality.
// CUDA inclusions are ideally confined to this implementation file.
//
// The notions from the StreamExecutor basically correspond to the CUDA streams
// programming model provided by the libcuda.so driver APIs, so we don't have
// to do much more than wrap the calls to the libraries appropriately.
#ifndef DLXNET_STREAM_EXECUTOR_GPU_GPU_KERNEL_H_
#define DLXNET_STREAM_EXECUTOR_GPU_GPU_KERNEL_H_

#include "dlxnet/stream_executor/opencl/ocl_driver.h"
#include "dlxnet/stream_executor/kernel_cache_config.h"
#include "dlxnet/stream_executor/platform/logging.h"
#include "dlxnet/stream_executor/platform/port.h"
#include "dlxnet/stream_executor/stream_executor_internal.h"

namespace stream_executor{
    // Wraps a GpuFunctionHandle to implement the platform-independent
    // KernelInterface.
    class OCLKernel : public internal::KernelInterface {
        public:
            OCLKernel()
                : gpu_function_(nullptr),
                arity_(0),
                preferred_cache_config_(KernelCacheConfig::kNoPreference) {}

            // Note that the function is unloaded when the module is unloaded, and the
            // module that the function is contained in is owned by the GpuExecutor.
            ~OCLKernel() override {}

            // As arity cannot be reflected upon using the CUDA API, the arity is
            // explicitly set during the GpuExecutor::GetKernel initialization process.
            void set_arity(unsigned arity) { arity_ = arity; }
            unsigned Arity() const override { return arity_; }

            // CUDA supports setting the preferred cache configuration of a
            // GpuFunctionHandle (more-or-less equivalent to a GpuKernel). We support this
            // via the below functions; users can set a preference, and that is applied
            // when the kernel is [lazy-]loaded (in GpuExecutor::Launch). The alternative
            // would be to load the kernel & set the preference when the user calls the
            // setter below; either approach is valid. Sets the current kernel cache
            // configuration preference.
            void SetPreferredCacheConfig(KernelCacheConfig config) override {
                preferred_cache_config_ = config;
            }

            // Returns the current kernel cache configuration preference.
            KernelCacheConfig GetPreferredCacheConfig() const override {
                return preferred_cache_config_;
            }

            // Returns the slot that the GpuFunctionHandle is stored within for this
            // object, for the CUDA API which wants to load into a GpuFunctionHandle*.
            GpuFunctionHandle* gpu_function_ptr() { return &gpu_function_; }

            // Returns the GpuFunctionHandle value for passing to the CUDA API.
            GpuFunctionHandle AsGpuFunctionHandle() const {
                DCHECK(gpu_function_() != nullptr);
                return gpu_function_;
            }

        private:
            GpuFunctionHandle gpu_function_;  // Wrapped CUDA kernel handle.
            unsigned arity_;  // Number of formal parameters the kernel takes.

            // Preferred (but not required) cache configuration for this kernel.
            KernelCacheConfig preferred_cache_config_;

    };

    // Given a platform-independent kernel datatype, returns the (const) internal
    // CUDA platform implementation pointer.
    inline const OCLKernel* AsGpuKernel(const KernelBase* kernel) {
        return static_cast<const OCLKernel*>(kernel->implementation());
    }

    // Given a platform-independent kernel datatype, returns the (non-const)
    // internal CUDA platform implementation pointer.
    inline OCLKernel* AsGpuKernel(KernelBase* kernel) {
        return static_cast<OCLKernel*>(kernel->implementation());
    }

}//namespace stream_executor

#endif
