#ifndef DLXNET_STREAM_EXECUTOR_OCL_OCL_DRIVER_H_
#define DLXNET_STREAM_EXECUTOR_OCL_OCL_DRIVER_H_
#include <vector>

#include "dlxnet/stream_executor/lib/status.h"
#include "dlxnet/stream_executor/gpu/gpu_types.h"


namespace stream_executor{
    using port::Status;
    namespace{
        cl::Platform platform_;
        std::vector<cl::Device> devices_;
        bool initialized_ = false;
    }// namespace

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

            // refers to stream executor, used to manager all objects in opencl context
            static Status CreateContext(cl::Device device, cl::Context* context);

            static Status CreateContext(int device_ordinal, cl::Context* context);

            // refers to stream in cuda context, used to run all command operations
            static bool CreateStream(cl::Context context,
                    cl::CommandQueue* command_queue);

            static Status InitEvent(cl::Context context, cl::Event* result);

            static bool GetProgramKernel(cl::Context context, cl::Program program,
                    const char* kernelname, cl::Kernel* kernel);
            static Status LoadText(cl::Context, absl::string_view fname,
                    GpuModuleHandle* module, const std::string build_options="");

            static Status LoadBin(cl::Context, absl::string_view fname,
                    GpuModuleHandle* module);

            static Status LaunchKernel(GpuContext context,
                    GpuFunctionHandle kernel, cl::NDRange gws, cl::NDRange lws,
                    GpuStreamHandle stream);

            static int GetDeviceCount(){
                return devices_.size();
            }

            void DestroyContext();

            static bool Initialized(){
                return initialized_;
            }
    };
}//namespace stream_executor


#endif
