#include "dlxnet/core/util/work_sharder.h"
#include "dlxnet/core/platform/logging.h"


namespace dlxnet{
    /* ABSL_CONST_INIT */ thread_local int per_thread_max_parallelism = 1000000;

    void SetPerThreadMaxParallelism(int max_parallelism) {
        CHECK_LE(0, max_parallelism);
        per_thread_max_parallelism = max_parallelism;
    }

    int GetPerThreadMaxParallelism() { return per_thread_max_parallelism; }
}
