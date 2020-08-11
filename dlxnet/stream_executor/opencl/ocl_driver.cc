#include <vector>
#include <fstream>
#include <string>
#include <cstring>

#include "dlxnet/stream_executor/platform/port.h"
#include "dlxnet/stream_executor/platform/logging.h"
#include "dlxnet/stream_executor/opencl/ocl_driver.h"
#include "dlxnet/stream_executor/lib/status_macros.h"
#include "dlxnet/core/lib/core/errors.h"

namespace stream_executor{
    namespace{
        cl::Platform platform_;
        std::vector<cl::Device> devices_;
        bool initialized_ = false;
    }// namespace

    GpuStreamHandle OCLDriver::default_stream_;

    Status OCLDriver::Init(){
        // To init platform is ok
        SE_RETURN_IF_ERROR(GetOrCreatePlatform());
        SE_RETURN_IF_ERROR(CreateDevicesList());
        CHECK(!Initialized())<<"Platform is Initialized by multiple times!";
        initialized_ = true;
        return Status::OK();
    }

    /*static*/ int OCLDriver::GetDeviceCount(){
        return devices_.size();
    }

    /*static*/ bool OCLDriver::Initialized(){
        return initialized_;
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
                "there may be "<<all_platforms.size()<<" opencl implementaions";
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
        void* ptr = clCreateBuffer(context(), CL_MEM_READ_WRITE, bytes, NULL, NULL);
        VLOG(2) << "allocated " << ptr << " for context " << &context
            << " of " << bytes << " bytes";
        return ptr;
    }

    void OCLDriver::DeviceDeallocate(GpuContext context, GpuDevicePtr gpu_ptr){
        clReleaseMemObject(gpu_ptr);
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

    /* static */ Status OCLDriver::DestroyEvent(GpuContext context,
            GpuEventHandle* event) {
        if ((*event)() == nullptr) {
            return Status(port::error::INVALID_ARGUMENT,
                    "input event cannot be null");
        }

        // ScopedActivateContext activated{context};
        // clReleaseEvent(*event);
        return Status::OK();
    }

    /* static */ Status OCLDriver::RecordEvent(GpuContext context,
            GpuEventHandle event,
            GpuStreamHandle stream) {
        // ScopedActivateContext activated{context};
        // RETURN_IF_CUDA_RES_ERROR(cuEventRecord(event, stream),
        // "Error recording CUDA event");
        return Status::OK();
    }

    /* static */ bool OCLDriver::WaitStreamOnEvent(GpuContext context,
            GpuStreamHandle stream, GpuEventHandle event) {
        // ScopedActivateContext activation(context);
        GpuStatus res = clWaitForEvents(1, &event());
        if (res != CL_SUCCESS) {
            LOG(ERROR) << "could not wait stream on event: " << res;
            return false;
        }

        return true;
    }

    /* static */ port::Status OCLDriver::QueryEvent(GpuContext context,
            GpuEventHandle event) {
        // ScopedActivateContext activated{context};
        // CUresult res = cuEventQuery(event);
        GpuStatus res = CL_SUCCESS;
        // if (res != CUDA_SUCCESS && res != CUDA_ERROR_NOT_READY) {
        // return port::Status(
        // port::error::INTERNAL,
        // absl::StrFormat("failed to query event: %s", ToString(res)));
        // }

        return Status::OK();
    }


    Status OCLDriver::LaunchKernel(GpuContext context,
            GpuFunctionHandle kernel, cl::NDRange gws, cl::NDRange lws,
            GpuStreamHandle stream){
        try{
            // enqueue
            stream.enqueueNDRangeKernel(kernel, cl::NullRange,
                    gws, lws);
        }catch(cl::Error error){
            return ::dlxnet::errors::Internal(error.what(), "(", error.err(), ")");
        }

        return Status::OK();
    }

    Status OCLDriver::GetDefaultStream(GpuContext context, GpuStreamHandle* stream){
        if(default_stream_()==nullptr){
            CreateStream(context, &default_stream_);
        }

        *stream = default_stream_;
        return Status::OK();
    }

    Status OCLDriver::SynchronousMemcpyD2H(GpuContext context, void* host_dst,
            GpuDevicePtr gpu_src_ptr, uint64 size){
        // make sure context is activated
        GpuStreamHandle stream;
        CreateStream(context, &stream);

        cl_int err;
        void* buffer_ptr = clEnqueueMapBuffer(stream(), gpu_src_ptr, CL_TRUE, CL_MAP_READ,
                0, size, 0, 0, 0, &err);
        memcpy(host_dst, buffer_ptr, size);

        clEnqueueUnmapMemObject(stream(), gpu_src_ptr, buffer_ptr, 0, 0, 0);
        if(err!=CL_SUCCESS){
            return ::dlxnet::errors::Internal("SynchronousMemcpyD2H Failed with error code: ", err);
        }

        return Status::OK();
    }

    Status OCLDriver::SynchronousMemcpyH2D(GpuContext context,
            GpuDevicePtr gpu_dst_ptr, const void* host_src, uint64 size){
        // make sure context is activated
        GpuStreamHandle stream;
        CreateStream(context, &stream);
        cl_int err;
        void* buffer_ptr = clEnqueueMapBuffer(stream(), gpu_dst_ptr, CL_TRUE, CL_MAP_WRITE,
                0, size, 0, 0, 0, &err);
        memcpy(buffer_ptr, host_src, size);
        clEnqueueUnmapMemObject(stream(), gpu_dst_ptr, buffer_ptr, 0, 0, 0);
        if(err!=CL_SUCCESS){
            return ::dlxnet::errors::Internal("SynchronousMemcpyH2D Failed with error code: ", err);
        }

        return Status::OK();
    }

    Status OCLDriver::SynchronousMemcpyD2D(GpuContext context,
            GpuDevicePtr gpu_dst_ptr, GpuDevicePtr gpu_src_ptr, uint64 size){
        // make sure context is activated
        GpuStreamHandle stream;
        SE_RETURN_IF_ERROR(GetDefaultStream(context, &stream));
        if(CL_SUCCESS==stream.enqueueCopyBuffer(cl::Buffer(gpu_src_ptr),
                    cl::Buffer(gpu_dst_ptr), 0, 0, size)){
            return Status::OK();
        }
        return ::dlxnet::errors::Internal("Failed to Sync memcpy from device to device");
    }

    /*static*/ bool OCLDriver::AsynchronousMemcpyD2H(GpuContext context, void* host_dst,
            GpuDevicePtr gpu_src, uint64 size,
            GpuStreamHandle stream){
        // if(CL_SUCCESS==stream.enqueueReadBuffer(cl::Buffer(gpu_src),
        // CL_FALSE, 0, size, host_dst)){
        // return true;
        // }
        // return false;
        cl_int err;
        void* buffer_ptr = clEnqueueMapBuffer(stream(), gpu_src, CL_TRUE, CL_MAP_READ,
                0, size, 0, 0, 0, &err);
        memcpy(host_dst, buffer_ptr, size);

        clEnqueueUnmapMemObject(stream(), gpu_src, buffer_ptr, 0, 0, 0);
        if(err!=CL_SUCCESS){
            return false;
        }

        return true;
    }

    /*static*/ bool OCLDriver::AsynchronousMemcpyH2D(GpuContext context, GpuDevicePtr gpu_dst,
            const void* host_src, uint64 size,
            GpuStreamHandle stream){
        // if(CL_SUCCESS==stream.enqueueWriteBuffer(cl::Buffer(gpu_dst),
        // CL_TRUE, 0, size, host_src)){
        // int host_tmp[4]={0};
        // stream.enqueueReadBuffer(cl::Buffer(gpu_dst),
        // CL_TRUE, 0, size, host_tmp);
        // return true;
        // }
        // return false;
        cl_int err;
        void* buffer_ptr = clEnqueueMapBuffer(stream(), gpu_dst, CL_TRUE, CL_MAP_WRITE,
                0, size, 0, 0, 0, &err);
        memcpy(buffer_ptr, host_src, size);
        clEnqueueUnmapMemObject(stream(), gpu_dst, buffer_ptr, 0, 0, 0);
        if(err!=CL_SUCCESS){
            return false;
        }

        return true;
    }

    /*static*/ bool OCLDriver::AsynchronousMemcpyD2D(GpuContext context, GpuDevicePtr gpu_dst,
            GpuDevicePtr gpu_src, uint64 size,
            GpuStreamHandle stream){
        if(CL_SUCCESS==stream.enqueueCopyBuffer(cl::Buffer(gpu_src),
                    cl::Buffer(gpu_dst), 0, 0, size)){
            return true;
        }
        return false;
    }
}//namespace stream_executor
