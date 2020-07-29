#ifndef DLXNET_CORE_COMMON_RUNTIME_PROCESS_UTIL_H_
#define DLXNET_CORE_COMMON_RUNTIME_PROCESS_UTIL_H_
#include "dlxnet/core/lib/core/threadpool.h"
#include "dlxnet/core/public/session_options.h"

namespace dlxnet{
    // Returns a process-wide ThreadPool for scheduling compute operations
    // using 'options'.  Caller does not take ownership over threadpool.
    thread::ThreadPool* ComputePool(const SessionOptions& options);

    // Returns the TF_NUM_INTEROP_THREADS environment value, or 0 if not specified.
    int32 NumInterOpThreadsFromEnvironment();

    // Returns the TF_NUM_INTRAOP_THREADS environment value, or 0 if not specified.
    int32 NumIntraOpThreadsFromEnvironment();

    // Returns the number of inter op threads specified in `options` or a default.
    // If no value or a negative value is specified in the provided options, then
    // the function returns the value defined in the TF_NUM_INTEROP_THREADS
    // environment variable. If neither a value is specified in the options or in
    // the environment, this function will return a reasonable default value based
    // on the number of schedulable CPUs, and any MKL and OpenMP configurations.
    int32 NumInterOpThreadsFromSessionOptions(const SessionOptions& options);

    // Creates a thread pool with number of inter op threads.
    thread::ThreadPool* NewThreadPoolFromSessionOptions(
            const SessionOptions& options);
} // namespace dlxnet


#endif
