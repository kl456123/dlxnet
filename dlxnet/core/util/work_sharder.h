#ifndef DLXNET_CORE_UTIL_WORK_SHARDER_H_
#define DLXNET_CORE_UTIL_WORK_SHARDER_H_

namespace dlxnet{
    // Each thread has an associated option to express the desired maximum
    // parallelism. Its default is a very large quantity.
    //
    // Within TF runtime, per-thread max parallelism affects Shard() and
    // intra-op parallelism. E.g., if SetPerThreadMaxParallelism(1) is
    // arranged to be called by a tf_compute thread, Shard() calls and
    // eigen device assignment happens in that thread afterwards becomes
    // single-threaded.
    void SetPerThreadMaxParallelism(int max_parallelism);
    int GetPerThreadMaxParallelism();
}

#endif
