#ifndef DLXNET_STREAM_EXECUTOR_EXECUTOR_CACHE_H_
#define DLXNET_STREAM_EXECUTOR_EXECUTOR_CACHE_H_

#include <functional>
#include <map>

#include "absl/synchronization/mutex.h"
#include "dlxnet/stream_executor/lib/status.h"
#include "dlxnet/stream_executor/lib/statusor.h"
#include "dlxnet/stream_executor/stream_executor.h"

namespace stream_executor {
    // Utility class to allow Platform objects to manage cached StreamExecutors.
    // Thread-safe.
    class ExecutorCache {
        public:
            ExecutorCache() {}

            // Looks up 'config' in the cache. Returns a pointer to the existing executor,
            // if already present, or creates it using 'factory', if it does not.
            // Factories may be executed concurrently for different device ordinals.
            typedef port::StatusOr<std::unique_ptr<StreamExecutor>> ExecutorFactory();
            port::StatusOr<StreamExecutor*> GetOrCreate(
                    const StreamExecutorConfig& config,
                    const std::function<ExecutorFactory>& factory);

            // Returns a pointer to the described executor (if one with a matching config
            // has been created), or a NOT_FOUND status.
            port::StatusOr<StreamExecutor*> Get(const StreamExecutorConfig& config);

            // Destroys all Executors and clears the cache.
            // Performs no synchronization with the executors - undefined behavior may
            // occur if any executors are active!
            void DestroyAllExecutors();

        private:
            // Each Entry contains zero or more cached executors for a device ordinal.
            struct Entry {
                ~Entry();

                // Mutex that guards the contents of each entry. The 'mutex_' of the
                // ExecutorCache class protects both the 'cache_' and the existence of each
                // Entry, but not the Entry's contents. 'configurations_mutex' protects the
                // contents of the entry after 'mutex_' has been dropped.
                absl::Mutex configurations_mutex;

                // Vector of cached {config, executor} pairs.
                std::vector<
                    std::pair<StreamExecutorConfig, std::unique_ptr<StreamExecutor>>>
                    configurations GUARDED_BY(configurations_mutex);
            };

            // Maps ordinal number to a list of cached executors for that ordinal.
            // We key off of ordinal (instead of just looking up all fields in the
            // StreamExecutorConfig) for a slight improvement in lookup time.
            absl::Mutex mutex_;
            std::map<int, Entry> cache_ GUARDED_BY(mutex_);

            SE_DISALLOW_COPY_AND_ASSIGN(ExecutorCache);
    };
}


#endif
