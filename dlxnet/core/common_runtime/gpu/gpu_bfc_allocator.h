#ifndef DLXNET_CORE_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_
#define DLXNET_CORE_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_
#include "dlxnet/core/common_runtime/bfc_allocator.h"
#include "dlxnet/core/common_runtime/gpu/gpu_mem_allocator.h"
#include "dlxnet/core/protobuf/config.pb.h"

namespace dlxnet{
    // A GPU memory allocator that implements a 'best-fit with coalescing'
    // algorithm.
    class GPUBFCAllocator : public BFCAllocator {
        public:
            GPUBFCAllocator(GPUMemAllocator* sub_allocator, size_t total_memory,
                    const string& name);
            GPUBFCAllocator(GPUMemAllocator* sub_allocator, size_t total_memory,
                    const GPUOptions& gpu_options, const string& name);
            ~GPUBFCAllocator() override {}

            TF_DISALLOW_COPY_AND_ASSIGN(GPUBFCAllocator);
        private:
            static bool GetAllowGrowthValue(const GPUOptions& gpu_options);
            static bool GetGarbageCollectionValue();
    };
}//namespace dlxnet

#endif
