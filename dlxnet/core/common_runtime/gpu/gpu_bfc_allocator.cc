#include "dlxnet/core/common_runtime/gpu/gpu_bfc_allocator.h"

namespace dlxnet{
    bool GPUBFCAllocator::GetAllowGrowthValue(const GPUOptions& gpu_options) {
        const char* force_allow_growth_string =
            std::getenv("TF_FORCE_GPU_ALLOW_GROWTH");
        if (force_allow_growth_string == nullptr) {
            return gpu_options.allow_growth();
        }

        if (strcmp("false", force_allow_growth_string) == 0) {
            if (gpu_options.allow_growth()) {
                LOG(WARNING)
                    << "Overriding allow_growth setting because the"
                    << " TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original"
                    << " config value was " << gpu_options.allow_growth() << ".";
            }
            return false;
        } else if (strcmp("true", force_allow_growth_string) == 0) {
            if (!gpu_options.allow_growth()) {
                LOG(WARNING)
                    << "Overriding allow_growth setting because the"
                    << " TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original"
                    << " config value was " << gpu_options.allow_growth() << ".";
            }
            return true;
        }

        LOG(ERROR)
            << "The TF_FORCE_GPU_ALLOW_GROWTH environment variable is set but could"
            << " not be parsed: \"" << force_allow_growth_string << "\". Valid"
            << " values are \"true\" or \"false\". Using original config value"
            << " of " << gpu_options.allow_growth() << ".";
        return gpu_options.allow_growth();
    }
    bool GPUBFCAllocator::GetGarbageCollectionValue() {
        const char* enable_gpu_garbage_collection =
            std::getenv("TF_ENABLE_GPU_GARBAGE_COLLECTION");
        if (enable_gpu_garbage_collection == nullptr) {
            // By default, turn on the memory garbage collection.
            return true;
        }
        if (strcmp("false", enable_gpu_garbage_collection) == 0) {
            return false;
        } else if (strcmp("true", enable_gpu_garbage_collection) == 0) {
            return true;
        }

        LOG(ERROR)
            << "The TF_ENABLE_GPU_GARBAGE_COLLECTION environment variable is set but"
            << " could not be parsed: \"" << enable_gpu_garbage_collection << "\"."
            << " Valid values are \"true\" or \"false\"."
            << " Using the default value \"true\".";
        return true;
    }

    GPUBFCAllocator::GPUBFCAllocator(GPUMemAllocator* sub_allocator,
            size_t total_memory, const string& name)
        : GPUBFCAllocator(sub_allocator, total_memory, GPUOptions(), name) {}

    GPUBFCAllocator::GPUBFCAllocator(GPUMemAllocator* sub_allocator,
            size_t total_memory,
            const GPUOptions& gpu_options,
            const string& name)
        : BFCAllocator(sub_allocator, total_memory,
                GPUBFCAllocator::GetAllowGrowthValue(gpu_options), name,
                GPUBFCAllocator::GetGarbageCollectionValue()) {}
}//namespace dlxnet
