#include "dlxnet/core/common_runtime/process_util.h"

#include "dlxnet/core/lib/core/threadpool.h"
#include "dlxnet/core/platform/types.h"
#include "dlxnet/core/platform/cpu_info.h"


namespace dlxnet{
    namespace{
        // Use environment setting if specified (init once)
        int32 GetEnvNumInterOpThreads() {
            static int32 env_num_threads = NumInterOpThreadsFromEnvironment();
            return env_num_threads;
        }

        int32 DefaultNumInterOpThreads() {
#ifndef __ANDROID__
            int32 env_num_threads = GetEnvNumInterOpThreads();
            if (env_num_threads > 0) {
                return env_num_threads;
            }

            // Default to the maximum parallelism for the current process.
            return port::MaxParallelism();
#else
            // Historically, -D__ANDROID__ resulted in the inter-op threadpool not being
            // used (regardless of what was chosen here); instead, all work was done on
            // the thread(s) calling Session::Run. That's no longer the case, but we'd
            // like to avoid suddenly higher concurrency and peak resource usage (for the
            // same device shape, graph, and options) versus prior versions - as best we
            // can:
            //
            //   - Single Session::Run (none concurrent), and default options:
            //     Behavior is mostly the same as before.
            //
            //   - Concurrent Session::Runs, and default options:
            //     Reduced concurrency versus before.
            //
            //   - Thread-pool size set explicitly (>1):
            //     Increased concurrency versus before.
            //
            // (We assume the first case is the most common)
            return 1;
#endif
        }

        static thread::ThreadPool* InitComputePool(const SessionOptions& options) {
            int32 inter_op_parallelism_threads =
                options.config.inter_op_parallelism_threads();
            if (inter_op_parallelism_threads == 0) {
                inter_op_parallelism_threads = DefaultNumInterOpThreads();
            }
            return new thread::ThreadPool(
                    Env::Default(), ThreadOptions(), "Compute", inter_op_parallelism_threads,
                    !options.config.experimental().disable_thread_spinning(),
                    /*allocator=*/nullptr);
        }
    } // namespace

    thread::ThreadPool* ComputePool(const SessionOptions& options) {
        static thread::ThreadPool* compute_pool = InitComputePool(options);
        return compute_pool;
    }

    int32 NumInterOpThreadsFromEnvironment() {
        int32 num;
        const char* val = std::getenv("TF_NUM_INTEROP_THREADS");
        return (val && strings::safe_strto32(val, &num)) ? num : 0;
    }

    int32 NumIntraOpThreadsFromEnvironment() {
        int32 num;
        const char* val = std::getenv("TF_NUM_INTRAOP_THREADS");
        return (val && strings::safe_strto32(val, &num)) ? num : 0;
    }

    int32 NumInterOpThreadsFromSessionOptions(const SessionOptions& options) {
        const int32 inter_op = options.config.inter_op_parallelism_threads();
        if (inter_op > 0) return inter_op;
        const int32 env_inter_op = GetEnvNumInterOpThreads();
        if (env_inter_op > 0) return env_inter_op;
        return DefaultNumInterOpThreads();
    }

    thread::ThreadPool* NewThreadPoolFromSessionOptions(
            const SessionOptions& options) {
        const int32 num_threads = NumInterOpThreadsFromSessionOptions(options);
        VLOG(1) << "Direct session inter op parallelism threads: " << num_threads;
        return new thread::ThreadPool(
                options.env, ThreadOptions(), "Compute", num_threads,
                !options.config.experimental().disable_thread_spinning(),
                /*allocator=*/nullptr);
    }
} // namespace dlxnet
