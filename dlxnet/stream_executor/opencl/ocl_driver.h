#ifndef DLXNET_STREAM_EXECUTOR_OCL_OCL_DRIVER_H_
#define DLXNET_STREAM_EXECUTOR_OCL_OCL_DRIVER_H_
#include <vector>

#include "dlxnet/stream_executor/lib/status.h"
#include "dlxnet/stream_executor/gpu/gpu_types.h"
#include "dlxnet/stream_executor/lib/statusor.h"


namespace stream_executor{
    using port::Status;

    class OCLDriver{
        public:
            static Status Init();
            static Status GetOrCreatePlatform(
                    cl::Platform* default_platform=nullptr);

            static Status GetDevice(int device_ordinal, cl::Device* default_device);
            static Status CreateDefaultDevice(cl::Device* default_device){
                return GetDevice(0, default_device);
            }

            static Status CreateDevicesList(std::vector<cl::Device>* all_devices=nullptr);

            static Status CreateDeviceDescription(int device_ordinal);

            static void* DeviceAllocate(GpuContext context, uint64 bytes);
            static void DeviceDeallocate(GpuContext context, GpuDevicePtr gpu_ptr);

            // refers to stream executor, used to manager all objects in opencl context
            static Status CreateContext(cl::Device device, cl::Context* context);

            static Status CreateContext(int device_ordinal, cl::Context* context);

            // refers to stream in cuda context, used to run all command operations
            static bool CreateStream(cl::Context context,
                    cl::CommandQueue* command_queue);

            static Status InitEvent(cl::Context context, GpuEventHandle* result);

            // Records that an event occurred when execution reaches the current point in
            // thestream via cuEventRecord.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1
            static Status RecordEvent(GpuContext context, GpuEventHandle event,
                    GpuStreamHandle stream);

            // Destroys *event and turns it into a nullptr. event may not be null, but
            // *event may be, via cuEventDestroy
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef
            static Status DestroyEvent(GpuContext context, GpuEventHandle* event);

            // Polls (without blocking) to determine the status of an event - pending or
            // complete (or an error status).
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef
            static port::Status QueryEvent(GpuContext context,
                    GpuEventHandle event);

            // Causes stream to wait for event to trigger before proceeding via
            // cuStreamWaitEvent.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#axzz334nAXAhM
            static bool WaitStreamOnEvent(GpuContext context, GpuStreamHandle stream,
                    GpuEventHandle event);

            static bool GetProgramKernel(cl::Context context, cl::Program program,
                    const char* kernelname, cl::Kernel* kernel);
            static Status LoadText(cl::Context, absl::string_view fname,
                    GpuModuleHandle* module, const std::string build_options="");

            static Status LoadBin(cl::Context, absl::string_view fname,
                    GpuModuleHandle* module);

            static Status LaunchKernel(GpuContext context,
                    GpuFunctionHandle kernel, cl::NDRange gws, cl::NDRange lws,
                    GpuStreamHandle stream);

            // -- Synchronous memcopies.

            static port::Status SynchronousMemcpyD2H(GpuContext context, void* host_dst,
                    GpuDevicePtr gpu_src, uint64 size);
            static port::Status SynchronousMemcpyH2D(GpuContext context,
                    GpuDevicePtr gpu_dst,
                    const void* host_src, uint64 size);
            static port::Status SynchronousMemcpyD2D(GpuContext context,
                    GpuDevicePtr gpu_dst,
                    GpuDevicePtr gpu_src, uint64 size);

            // -- Asynchronous memcopies.

            static bool AsynchronousMemcpyD2H(GpuContext context, void* host_dst,
                    GpuDevicePtr gpu_src, uint64 size,
                    GpuStreamHandle stream);
            static bool AsynchronousMemcpyH2D(GpuContext context, GpuDevicePtr gpu_dst,
                    const void* host_src, uint64 size,
                    GpuStreamHandle stream);
            static bool AsynchronousMemcpyD2D(GpuContext context, GpuDevicePtr gpu_dst,
                    GpuDevicePtr gpu_src, uint64 size,
                    GpuStreamHandle stream);

            static int GetDeviceCount();

            void DestroyContext();

            static bool Initialized();

            static Status GetDefaultStream(GpuContext context, GpuStreamHandle* stream);
        private:
            static GpuStreamHandle default_stream_;
    };
}//namespace stream_executor


#endif
