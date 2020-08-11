#ifndef DLXNET_STREAM_EXECUTOR_GPU_GPU_TYPES_H_
#define DLXNET_STREAM_EXECUTOR_GPU_GPU_TYPES_H_
// cl headers
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

namespace stream_executor{
    using GpuDeviceHandle = cl::Device;
    using GpuContext = cl::Context;
    using GpuStreamHandle = cl::CommandQueue;
    using GpuFunctionHandle = cl::Kernel;
    using GpuStatus = cl_int;
    using GpuModuleHandle = cl::Program;
    using GpuEventHandle = cl::Event;
    using GpuDevicePtr = cl_mem;
}


#endif
