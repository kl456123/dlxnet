#include "dlxnet/core/framework/allocator.h"
#include "dlxnet/core/framework/allocator_registry.h"
#include "dlxnet/core/platform/mem.h"


namespace dlxnet{
    // If true, cpu allocator collects more stats.
    static bool cpu_allocator_collect_stats = false;

    void EnableCPUAllocatorStats(bool enable) {
        cpu_allocator_collect_stats = enable;
    }
    bool CPUAllocatorStatsEnabled() { return cpu_allocator_collect_stats; }


    namespace{
        class CPUAllocator:public Allocator{
            public:
                CPUAllocator(){}
                ~CPUAllocator()override{}
                std::string Name()override{return "cpu";}

                void* AllocateRaw(size_t alignment, size_t num_bytes) override {
                    void* p = port::AlignedMalloc(num_bytes, alignment);
                    if (cpu_allocator_collect_stats) {
                        const std::size_t alloc_size = port::MallocExtension_GetAllocatedSize(p);
                        ++stats_.num_allocs;
                        stats_.bytes_in_use += alloc_size;
                        stats_.peak_bytes_in_use =
                            std::max<int64_t>(stats_.peak_bytes_in_use, stats_.bytes_in_use);
                        stats_.largest_alloc_size =
                            std::max<int64_t>(stats_.largest_alloc_size, alloc_size);
                    }
                    return p;
                }

                void DeallocateRaw(void* ptr) override {
                    if (cpu_allocator_collect_stats) {
                        const std::size_t alloc_size =
                            port::MallocExtension_GetAllocatedSize(ptr);
                        stats_.bytes_in_use -= alloc_size;
                    }
                    port::AlignedFree(ptr);
                }

                absl::optional<AllocatorStats> GetStats()override{
                    return stats_;
                }

                void ClearStats() override {
                    stats_.num_allocs = 0;
                    stats_.peak_bytes_in_use = stats_.bytes_in_use;
                    stats_.largest_alloc_size = 0;
                }

            private:
                AllocatorStats stats_;
                TF_DISALLOW_COPY_AND_ASSIGN(CPUAllocator);
        };

        class CPUAllocatorFactory : public AllocatorFactory {
            public:
                Allocator* CreateAllocator() override { return new CPUAllocator; }
                SubAllocator* CreateSubAllocator(int numa_node)override {
                    return new CPUSubAllocator(new CPUAllocator);
                }
            private:
                class CPUSubAllocator : public SubAllocator {
                    public:
                        explicit CPUSubAllocator(CPUAllocator* cpu_allocator)
                            : SubAllocator({}, {}), cpu_allocator_(cpu_allocator) {}

                        void* Alloc(size_t alignment, size_t num_bytes) override {
                            return cpu_allocator_->AllocateRaw(alignment, num_bytes);
                        }

                        void Free(void* ptr, size_t num_bytes) override {
                            cpu_allocator_->DeallocateRaw(ptr);
                        }

                    private:
                        CPUAllocator* cpu_allocator_;
                };
        };

        REGISTER_MEM_ALLOCATOR("DefaultCPUAllocator", 100, CPUAllocatorFactory);
    }// namespace
}// namespace dlxnet
