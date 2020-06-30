#ifndef DLXNET_CORE_COMMON_RUNTIME_GPU_GPU_INIT_H_
#define DLXNET_CORE_COMMON_RUNTIME_GPU_GPU_INIT_H_
#include "dlxnet/core/lib/core/status.h"


namespace stream_executor {
  class Platform;
}  // namespace stream_executor


namespace dlxnet{
  // Initializes the GPU platform and returns OK if the GPU
  // platform could be initialized.
  Status ValidateGPUMachineManager();

  // Returns the GPU machine manager singleton, creating it and
  // initializing the GPUs on the machine if needed the first time it is
  // called.  Must only be called when there is a valid GPU environment
  // in the process (e.g., ValidateGPUMachineManager() returns OK).
  stream_executor::Platform* GPUMachineManager();

  // Returns the string describing the name of the GPU platform in use.
  // This value is "CUDA" by default, and
  // "ROCM" when TF is built with `--config==rocm`
  string GpuPlatformName();
} // namespace dlxnet



#endif
