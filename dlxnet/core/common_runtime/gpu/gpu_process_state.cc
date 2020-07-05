#include "dlxnet/core/common_runtime/gpu/gpu_process_state.h"
#include "dlxnet/core/platform/logging.h"
#include "dlxnet/core/common_runtime/gpu/gpu_cudamalloc_allocator.h"
#include "dlxnet/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "dlxnet/core/common_runtime/gpu/gpu_id_manager.h"
#include "dlxnet/core/common_runtime/gpu/gpu_id_utils.h"
#include "dlxnet/core/platform/macros.h"
#include "dlxnet/core/lib/core/status.h"
#include "dlxnet/core/lib/strings/strcat.h"


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
    int GPUProcessState::BusIdForGPU(TfGpuId tf_gpu_id) {
        // Return the NUMA node associated with the GPU's StreamExecutor.
        se::StreamExecutor* se =
            GpuIdUtil::ExecutorForTfGpuId(tf_gpu_id).ValueOrDie();
        int numa_node = se->GetDeviceDescription().numa_node();
        // bus_id must be non-negative.  If the numa_node is not known,
        // use 0.
        return numa_node >= 0 ? numa_node : 0;
    }

    Allocator* GPUProcessState::GetGpuHostAllocator(int numa_node) {
        CHECK(process_state_);
        if (!HasGPUDevice() ||
                !process_state_->ProcessState::FLAGS_brain_mem_reg_gpu_dma) {
            return process_state_->GetCPUAllocator(numa_node);
        }
        if (numa_node == port::kNUMANoAffinity) {
            numa_node = 0;
        }
        LOG(FATAL)<< "Cannot create host allocator by gpu ps state, use cpu ps instead";
        return nullptr;
    }

    Allocator* GPUProcessState::GetGPUAllocator(const GPUOptions& options,
            TfGpuId tf_gpu_id, size_t total_bytes) {
        CHECK(process_state_);

        const string& allocator_type = options.allocator_type();
        mutex_lock lock(mu_);

        if (tf_gpu_id >= static_cast<int64>(gpu_allocators_.size())) {
            gpu_allocators_.resize(tf_gpu_id + 1);
        }

        AllocatorParts& allocator_parts = gpu_allocators_[tf_gpu_id];
        if (allocator_parts.allocator == nullptr) {
            // Validate allocator types.
            if (!allocator_type.empty() && allocator_type != "BFC") {
                LOG(ERROR) << "Invalid allocator type: " << allocator_type;
                return nullptr;
            }
            // create gpu allocator
            PlatformGpuId platform_gpu_id;
            TF_CHECK_OK(GpuIdManager::TfToPlatformGpuId(tf_gpu_id, &platform_gpu_id));
            int bus_id = BusIdForGPU(tf_gpu_id);
            DCHECK_GE(bus_id, 0);
            while (bus_id >= gpu_visitors_.size()) {
                gpu_visitors_.push_back({});
            }

            GPUMemAllocator* sub_allocator = new GPUMemAllocator(
                    GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
                    platform_gpu_id,
                    (options.per_process_gpu_memory_fraction() > 1.0 ||
                     options.experimental().use_unified_memory()),
                    gpu_visitors_[bus_id], {});
            GPUBFCAllocator* gpu_bfc_allocator =
                new GPUBFCAllocator(sub_allocator, total_bytes, options,
                        strings::StrCat("GPU_", tf_gpu_id, "_bfc"));
            Allocator* gpu_allocator = gpu_bfc_allocator;

            if (useCudaMallocAllocator()) {
                LOG(INFO) << "Using CUDA malloc allocator for GPU.";
                // If true, passes all allocation requests through to cudaMalloc
                // useful for doing memory debugging with tools like cuda-memcheck
                // **WARNING** probably will not work in a multi-gpu scenario
                gpu_allocator =
                    new GPUcudaMallocAllocator(gpu_allocator, platform_gpu_id);
            }
            // records allocator
            Allocator* recording_allocator = nullptr;
            if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types) {
                // do nothing with recording at present
                recording_allocator = gpu_allocator;
            }
            allocator_parts = {std::unique_ptr<Allocator>(gpu_allocator),
                gpu_bfc_allocator, sub_allocator,
                std::unique_ptr<Allocator>(recording_allocator)};
        }

        if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types) {
            return allocator_parts.recording_allocator.get();
        } else {
            return allocator_parts.allocator.get();
        }
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
