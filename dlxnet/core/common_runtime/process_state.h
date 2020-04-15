#ifndef DLXNET_CORE_COMMON_RUNTIME_PROCESS_STATE_H_
#define DLXNET_CORE_COMMON_RUNTIME_PROCESS_STATE_H_
#include "dlxnet/core/framework/allocator_registry.h"
#include "dlxnet/core/framework/allocator.h"
#include "dlxnet/core/platform/macros.h"

namespace dlxnet{
    // Singleton that manages per-process state, e.g. allocation of
    // shared resources.
    class ProcessState : public ProcessStateInterface {
        public:
            static ProcessState* singleton();
            // If NUMA Allocators are desired, call this before calling any
            // Allocator accessor.
            void EnableNUMA() { numa_enabled_ = true; }

            // Returns the one CPUAllocator used for the given numa_node.
            // Treats numa_node == kNUMANoAffinity as numa_node == 0.
            Allocator* GetCPUAllocator(int numa_node) override;

            // Registers alloc visitor for the CPU allocator(s).
            // REQUIRES: must be called before GetCPUAllocator.
            void AddCPUAllocVisitor(SubAllocator::Visitor v);

            // Registers free visitor for the CPU allocator(s).
            // REQUIRES: must be called before GetCPUAllocator.
            void AddCPUFreeVisitor(SubAllocator::Visitor v);

        protected:
            ProcessState();
            virtual ~ProcessState() {}
            friend class GPUProcessState;

            // If these flags need to be runtime configurable consider adding
            // them to ConfigProto.
            static const bool FLAGS_brain_mem_reg_gpu_dma = true;
            static const bool FLAGS_brain_gpu_record_mem_types = false;

            static ProcessState* instance_;
            bool numa_enabled_;

            mutex mu_;

            // Indexed by numa_node.  If we want numa-specific allocators AND a
            // non-specific allocator, maybe should index by numa_node+1.
            std::vector<Allocator*> cpu_allocators_ GUARDED_BY(mu_);
            std::vector<SubAllocator::Visitor> cpu_alloc_visitors_ GUARDED_BY(mu_);
            std::vector<SubAllocator::Visitor> cpu_free_visitors_ GUARDED_BY(mu_);

            // Optional RecordingAllocators that wrap the corresponding
            // Allocators for runtime attribute use analysis.
            // MDMap mem_desc_map_;
            std::vector<Allocator*> cpu_al_ GUARDED_BY(mu_);
    };

}//namespace dlxnet

#endif
