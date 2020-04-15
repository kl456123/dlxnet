#include "dlxnet/core/common_runtime/process_state.h"
#include "dlxnet/core/common_runtime/pool_allocator.h"
#include "dlxnet/core/common_runtime/bfc_allocator.h"
#include "dlxnet/core/framework/log_memory.h"
#include "dlxnet/core/framework/tracking_allocator.h"
#include "dlxnet/core/util/env_var.h"


namespace dlxnet{
    /*static*/ ProcessState* ProcessState::singleton() {
        static ProcessState* instance = new ProcessState;
        // static std::once_flag f;
        // std::call_once(f, []() {
        // AllocatorFactoryRegistry::singleton()->process_state_ = instance;
        // });

        return instance;
    }

    ProcessState::ProcessState() : numa_enabled_(false) {}

    Allocator* ProcessState::GetCPUAllocator(int numa_node) {
        if (!numa_enabled_ || numa_node == port::kNUMANoAffinity) numa_node = 0;
        mutex_lock lock(mu_);
        while (cpu_allocators_.size() <= static_cast<size_t>(numa_node)) {
            // If visitors have been defined we need an Allocator built from
            // a SubAllocator.  Prefer BFCAllocator, but fall back to PoolAllocator
            // depending on env var setting.
            const bool alloc_visitors_defined =
                (!cpu_alloc_visitors_.empty() || !cpu_free_visitors_.empty());
            bool use_bfc_allocator = false;
            Status status = ReadBoolFromEnvVar(
                    "TF_CPU_ALLOCATOR_USE_BFC", alloc_visitors_defined, &use_bfc_allocator);
            if (!status.ok()) {
                LOG(ERROR) << "GetCPUAllocator: " << status.error_message();
            }

            // original sub_allocator
            Allocator* allocator=nullptr;
            SubAllocator* sub_allocator =
                (numa_enabled_ || alloc_visitors_defined || use_bfc_allocator)
                ? new BasicCPUAllocator(
                        numa_enabled_ ? numa_node : port::kNUMANoAffinity,
                        cpu_alloc_visitors_, cpu_free_visitors_)
                : nullptr;

            // check if use bfc or not
            if(use_bfc_allocator){
                // TODO(reedwm): evaluate whether 64GB by default is the best choice.
                int64 cpu_mem_limit_in_mb = -1;
                Status status = ReadInt64FromEnvVar("TF_CPU_BFC_MEM_LIMIT_IN_MB",
                        1LL << 16 /*64GB max by default*/,
                        &cpu_mem_limit_in_mb);
                if (!status.ok()) {
                    LOG(ERROR) << "GetCPUAllocator: " << status.error_message();
                }
                int64 cpu_mem_limit = cpu_mem_limit_in_mb * (1LL << 20);
                DCHECK(sub_allocator);
                allocator =
                    new BFCAllocator(sub_allocator, cpu_mem_limit, true /*allow_growth*/,
                            "bfc_cpu_allocator_for_gpu" /*name*/);
                VLOG(2) << "Using BFCAllocator with memory limit of "
                    << cpu_mem_limit_in_mb << " MB for ProcessState CPU allocator";
            }else if(sub_allocator){
                DCHECK(sub_allocator);
                allocator =
                    new PoolAllocator(100 /*pool_size_limit*/, true /*auto_resize*/,
                            sub_allocator, "cpu_pool");
                VLOG(2) << "Using PoolAllocator for ProcessState CPU allocator "
                    << "numa_enabled_=" << numa_enabled_
                    << " numa_node=" << numa_node;
            }else{
                // original allocator
                DCHECK(!sub_allocator);
                allocator = cpu_allocator_base();
            }

            if (LogMemory::IsEnabled() && !allocator->TracksAllocationSizes()) {
                // Wrap the allocator to track allocation ids for better logging
                // at the cost of performance.
                allocator = new TrackingAllocator(allocator, true);
            }

            cpu_allocators_.push_back(allocator);
            if (!sub_allocator) {
                DCHECK(cpu_alloc_visitors_.empty() && cpu_free_visitors_.empty());
            }
        }
        return cpu_allocators_[numa_node];
    }
}//namespace dlxnet
