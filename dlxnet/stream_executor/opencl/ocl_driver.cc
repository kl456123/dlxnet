#include <vector>

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
            return ::dlxnet::errors::Internal("Multiple platforms found, "
                    "there may be",all_platforms.size(),"opencl implementaions");
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
        context = new cl::Context({device});
        return Status::OK();
    }

    Status OCLDriver::CreateContext(int device_ordinal, cl::Context* context){
        CHECK_GE(devices_.size(), device_ordinal);
        return CreateContext(devices_[device_ordinal], context);
    }

    Status OCLDriver::CreateCommandQueue(cl::Context context,
            cl::Device device, cl::CommandQueue* command_queue){
        command_queue = new cl::CommandQueue(context, device);
        return Status::OK();
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
}//namespace stream_executor
