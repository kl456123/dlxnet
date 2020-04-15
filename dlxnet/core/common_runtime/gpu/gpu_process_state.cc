#include "dlxnet/core/common_runtime/gpu/gpu_process_state.h"
#include "dlxnet/core/platform/logging.h"


namespace dlxnet{
    namespace {

        bool useCudaMallocAllocator() {
            const char* debug_allocator_str = std::getenv("TF_GPU_ALLOCATOR");
            return debug_allocator_str != nullptr &&
                std::strcmp(debug_allocator_str, "cuda_malloc") == 0;
        }
    }//namespace
    /*static*/ GPUProcessState* GPUProcessState::singleton(GPUProcessState* ps) {
        static GPUProcessState* instance = ps ? ps : new GPUProcessState;
        DCHECK((!ps) || (ps == instance))
            << "Multiple calls to GPUProcessState with non-null ps";
        return instance;
    }

    GPUProcessState::GPUProcessState() : gpu_device_enabled_(false) {
        process_state_ = ProcessState::singleton();
    }
    int GPUProcessState::BusIdForGPU(TfGpuId tf_gpu_id) {}

    Allocator* GPUProcessState::GetGpuHostAllocator(int numa_node) {
        CHECK(process_state_);
        if (!HasGPUDevice() ||
                !process_state_->ProcessState::FLAGS_brain_mem_reg_gpu_dma) {
            return process_state_->GetCPUAllocator(numa_node);
        }
        if (numa_node == port::kNUMANoAffinity) {
            numa_node = 0;
        }
        return nullptr;
    }

    Allocator* GPUProcessState::GetGPUAllocator(const GPUOptions& options,
            TfGpuId tf_gpu_id,
            size_t total_bytes) {
    }

    void GPUProcessState::AddGPUAllocVisitor(int bus_id,
            const SubAllocator::Visitor& visitor) {
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
        (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
        mutex_lock lock(mu_);
        CHECK(gpu_allocators_.empty())  // Crash OK
            << "AddGPUAllocVisitor must be called before "
            "first call to GetGPUAllocator.";
        DCHECK_GE(bus_id, 0);
        while (bus_id >= static_cast<int64>(gpu_visitors_.size())) {
            gpu_visitors_.push_back(std::vector<SubAllocator::Visitor>());
        }
        gpu_visitors_[bus_id].push_back(visitor);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    }

    void GPUProcessState::AddGpuHostAllocVisitor(
            int numa_node, const SubAllocator::Visitor& visitor) {
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
        (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
        mutex_lock lock(mu_);
        CHECK(gpu_host_allocators_.empty())  // Crash OK
            << "AddGpuHostAllocVisitor must be called before "
            "first call to GetGpuHostAllocator.";
        while (numa_node >= static_cast<int64>(gpu_host_alloc_visitors_.size())) {
            gpu_host_alloc_visitors_.push_back(std::vector<SubAllocator::Visitor>());
        }
        gpu_host_alloc_visitors_[numa_node].push_back(visitor);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    }

    void GPUProcessState::AddGpuHostFreeVisitor(
            int numa_node, const SubAllocator::Visitor& visitor) {
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
        (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
        mutex_lock lock(mu_);
        CHECK(gpu_host_allocators_.empty())  // Crash OK
            << "AddGpuHostFreeVisitor must be called before "
            "first call to GetGpuHostAllocator.";
        while (numa_node >= static_cast<int64>(gpu_host_free_visitors_.size())) {
            gpu_host_free_visitors_.push_back(std::vector<SubAllocator::Visitor>());
        }
        gpu_host_free_visitors_[numa_node].push_back(visitor);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    }



}//namespace dlxnet
