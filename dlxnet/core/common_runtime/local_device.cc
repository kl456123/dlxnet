#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "dlxnet/core/common_runtime/local_device.h"
#include "dlxnet/core/framework/allocator.h"
#include "dlxnet/core/lib/core/threadpool.h"
#include "dlxnet/core/platform/cpu_info.h"

namespace dlxnet{
    struct LocalDevice::EigenThreadPoolInfo {
        // Wrapper so we can provide the CPUAllocator to Eigen for use
        // when ops need extra tmp memory.
        class EigenAllocator : public Eigen::Allocator {
            public:
                explicit EigenAllocator(dlxnet::Allocator* a) : allocator_(a) {}
                void* allocate(size_t num_bytes) const override {
                    return allocator_->AllocateRaw(64, num_bytes);
                }
                void deallocate(void* buffer) const override {
                    allocator_->DeallocateRaw(buffer);
                }
                dlxnet::Allocator* allocator_;
        };

        ~EigenThreadPoolInfo() {
            eigen_device_.reset();
        }

        explicit EigenThreadPoolInfo(const SessionOptions& options, int numa_node,
                Allocator* allocator) {
            // Use session setting if specified.
            int32 intra_op_parallelism_threads =
                options.config.intra_op_parallelism_threads();
            // If no session setting, use environment setting.
            if (intra_op_parallelism_threads == 0) {
                intra_op_parallelism_threads = port::MaxParallelism(numa_node);
            }
            ThreadOptions thread_opts;
            thread_opts.numa_node = numa_node;
            eigen_threadpool_.reset(new thread::ThreadPool(
                        options.env, thread_opts, strings::StrCat("numa_", numa_node, "_Eigen"),
                        intra_op_parallelism_threads,
                        !options.config.experimental().disable_thread_spinning(),
                        /*allocator=*/nullptr));
            Eigen::ThreadPoolInterface* threadpool =
                eigen_threadpool_->AsEigenThreadPool();
            if (allocator != nullptr) {
                eigen_allocator_.reset(new EigenAllocator(allocator));
            }
            eigen_device_.reset(new Eigen::ThreadPoolDevice(
                        threadpool, intra_op_parallelism_threads, eigen_allocator_.get()));
        }

        std::unique_ptr<thread::ThreadPool> eigen_threadpool_;
        std::unique_ptr<Eigen::ThreadPoolDevice> eigen_device_;
        std::unique_ptr<EigenAllocator> eigen_allocator_;
    };

    LocalDevice::LocalDevice(const SessionOptions& options,
            const DeviceAttributes& attributes)
        : Device(options.env, attributes), owned_tp_info_(nullptr){
            // Each LocalDevice owns a separate ThreadPoolDevice for numerical
            // computations.
            // TODO(tucker): NUMA for these too?
            owned_tp_info_.reset(new LocalDevice::EigenThreadPoolInfo(
                        options, port::kNUMANoAffinity, nullptr));
            LocalDevice::EigenThreadPoolInfo* tp_info;
            tp_info = owned_tp_info_.get();
            set_eigen_cpu_device(tp_info->eigen_device_.get());
            set_eigen_cpu_thread_pool(tp_info->eigen_threadpool_.get());
        }

    LocalDevice::~LocalDevice() {}
}
