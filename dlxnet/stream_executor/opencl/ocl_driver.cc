#include <vector>
#include <fstream>
#include <string>

#include "dlxnet/stream_executor/platform/port.h"
#include "dlxnet/stream_executor/opencl/ocl_driver.h"
#include "dlxnet/stream_executor/platform/logging.h"
#include "dlxnet/stream_executor/lib/status_macros.h"
#include "dlxnet/core/lib/core/errors.h"

namespace stream_executor{
    Status OCLDriver::Init(){
        // To init platform is ok
        SE_RETURN_IF_ERROR(GetOrCreatePlatform());
        SE_RETURN_IF_ERROR(CreateDevicesList());
        CHECK(!Initialized())<<"Platform is Initialized by multiple times!";
        initialized_ = true;
        return Status::OK();
    }

    Status OCLDriver::CreateDevicesList(std::vector<cl::Device>* devices){
        std::vector<cl::Device> all_devices;
        platform_.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
        devices_ = all_devices;

        if(devices){
            *devices = devices_;
        }
        return Status::OK();
    }

    Status OCLDriver::GetOrCreatePlatform(
            cl::Platform* default_platform){
        if(Initialized()){
            VLOG(1)<<"Get Platform from cache";
            *default_platform = platform_;
            return Status::OK();
        }
        VLOG(1)<<"Create Platform";
        std::vector<cl::Platform> all_platforms;
        cl::Platform::get(&all_platforms);
        if (all_platforms.size()==0) {
            return ::dlxnet::errors::NotFound("No platforms found. Check OpenCL installation!");
        }

        if(all_platforms.size()>1){
            // just log warnning instead of return error
            LOG(WARNING)<<"Multiple platforms found, "
                "there may be"<<all_platforms.size()<<"opencl implementaions";
        }
        platform_ = all_platforms[0];
        if(default_platform!=nullptr){
            *default_platform=platform_;
        }
        return Status::OK();
    }

    Status OCLDriver::GetDevice(int device_ordinal, cl::Device* device){
        CHECK(Initialized())<<"Platform is uninitialized, please try it first!";

        std::vector<cl::Device>& all_devices = devices_;
        platform_.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
        if(all_devices.size()==0){
            return ::dlxnet::errors::NotFound("No Devices found. Check OpenCL devices!");
        }
        if(all_devices.size()<=device_ordinal){
            return ::dlxnet::errors::InvalidArgument("Given device_ordinal is larger than the num of devices ",
                    all_devices.size());
        }

        *device = all_devices[device_ordinal];

        return Status::OK();
    }

    Status OCLDriver::CreateContext(cl::Device device, cl::Context* context){
        *context = cl::Context({device});
        return Status::OK();
    }

    Status OCLDriver::CreateContext(int device_ordinal, cl::Context* context){
        CHECK_GE(devices_.size(), device_ordinal);
        return CreateContext(devices_[device_ordinal], context);
    }

    bool OCLDriver::CreateStream(cl::Context context,
            cl::CommandQueue* command_queue){
        cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
        *command_queue = cl::CommandQueue(context, device);
        return true;
    }

    Status OCLDriver::CreateDeviceDescription(int device_ordinal){
        cl::Device device = devices_[device_ordinal];
        LOG(INFO)<<"Using device: "<<device.getInfo<CL_DEVICE_NAME>();
        int GPUComputeUnits;
        device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &GPUComputeUnits);
        LOG(INFO)<<"ComputeUnits: "<<GPUComputeUnits;
        LOG(INFO)<<"Max work group size: "<<device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        return Status::OK();
    }

    void OCLDriver::DestroyContext(){
    }

    void* OCLDriver::DeviceAllocate(GpuContext context, uint64 bytes){
        if(bytes==0){
            return nullptr;
        }
        cl::Buffer* buffer = new cl::Buffer(context, CL_MEM_READ_WRITE, bytes);
        void* ptr  = reinterpret_cast<void*>(buffer);
        VLOG(2) << "allocated " << ptr << " for context " << &context
            << " of " << bytes << " bytes";
        return ptr;
    }

    bool OCLDriver::GetProgramKernel(cl::Context context, cl::Program program,
            const char* kernelname, cl::Kernel* kernel){
        // make sure context is activated now

        // check args
        CHECK(program() != nullptr && kernelname != nullptr);

        // check error
        *kernel = cl::Kernel(program, kernelname);
        return true;
    }

    Status OCLDriver::LoadText(cl::Context context, absl::string_view fname,
            GpuModuleHandle* module, const std::string build_options){
        std::ifstream sourceFile(std::string(fname.data(), fname.size()));
        if(sourceFile.fail()){
            return ::dlxnet::errors::NotFound(fname, " Cannot be found");
        }
        std::string sourceCode(
                std::istreambuf_iterator<char>(sourceFile),
                (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(),
                    sourceCode.length()+1));

        // Make program of the source code in the context
        cl::Program program = cl::Program(context, source);

        VECTOR_CLASS<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        // Build program for these specific devices
        try{
            program.build(devices, build_options.c_str());
        } catch(cl::Error error) {
            if(error.err() == CL_BUILD_PROGRAM_FAILURE) {
                return ::dlxnet::errors::Internal("Build error log: ",
                        program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]));
            }
            return ::dlxnet::errors::Internal("Error happened in Build Program!");
        }

        *module = program;
        return Status::OK();
    }

    Status OCLDriver::LoadBin(cl::Context, absl::string_view fname,
            GpuModuleHandle* module){
        return Status::OK();
    }

    Status OCLDriver::InitEvent(cl::Context context, cl::Event* result){
        cl::Event event;
        *result = event;
        return Status::OK();
    }

    Status OCLDriver::LaunchKernel(GpuContext context,
            GpuFunctionHandle kernel, cl::NDRange gws, cl::NDRange lws,
            GpuStreamHandle stream){
        // enqueue
        stream.enqueueNDRangeKernel(kernel, cl::NullRange,
                gws, lws);
    }
}//namespace stream_executor
