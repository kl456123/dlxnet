#include "dlxnet/stream_executor/opencl/ocl_gpu_executor.h"
#include "dlxnet/stream_executor/opencl/ocl_driver.h"

namespace stream_executor{

    OCLExecutor::~OCLExecutor(){
        // note that all c++ api of opencl will release when recount == 0
    }

    port::Status OCLExecutor::Init(int device_ordinal,
            DeviceOptions device_options){
        device_ordinal_ = device_ordinal;
        // device

        auto status = OCLDriver::GetDevice(device_ordinal_, &device_);
        if (!status.ok()) {
            return status;
        }

        // context
        status = OCLDriver::CreateContext(device_, &context_);

        if (!status.ok()) {
            return status;
        }
        return status;
    }

    port::StatusOr<std::unique_ptr<DeviceDescription>>
        OCLExecutor::CreateDeviceDescription(int device_ordinal) {
            GpuDeviceHandle device;
            auto status = OCLDriver::GetDevice(device_ordinal, &device);
            if (!status.ok()) {
                return status;
            }

            internal::DeviceDescriptionBuilder builder;
            // set by builder
            return builder.Build();
        }

    // memory manager
    DeviceMemoryBase OCLExecutor::Allocate(uint64 size, int64 memory_space){
        CHECK_EQ(memory_space, 0);
        return DeviceMemoryBase(OCLDriver::DeviceAllocate(context_, size), size);
    }

    void OCLExecutor::Deallocate(DeviceMemoryBase *mem){
    }

    // implementation
    std::unique_ptr<internal::TimerInterface> OCLExecutor::GetTimerImplementation(){
    }

    std::unique_ptr<internal::StreamInterface> OCLExecutor::GetStreamImplementation(){
    }

    std::unique_ptr<internal::KernelInterface> OCLExecutor::CreateKernelImplementation(){
    }

    std::unique_ptr<internal::EventInterface> OCLExecutor::CreateEventImplementation(){
    }

    // event
    port::Status OCLExecutor::WaitForEvent(Stream* stream, Event* event){
    }

    port::Status OCLExecutor::RecordEvent(Stream* stream, Event* event){
    }

    port::Status OCLExecutor::AllocateEvent(Event* event){}

    port::Status OCLExecutor::DeallocateEvent(Event* event){}
    Event::Status OCLExecutor::PollForEventStatus(Event* event) {}

    port::Status OCLExecutor::BlockHostUntilDone(Stream* stream){
    }

    // stream
    bool OCLExecutor::AllocateStream(Stream* stream){
    }

    void OCLExecutor::DeallocateStream(Stream* stream){
    }

    // timer
    bool OCLExecutor::AllocateTimer(Timer* timer){}

    void OCLExecutor::DeallocateTimer(Timer* timer){}

    bool OCLExecutor::StartTimer(Stream* stream, Timer* timer){}

    bool OCLExecutor::StopTimer(Stream* stream, Timer* timer){
    }

    // copy functions
    port::Status OCLExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
            const void* host_src, uint64 size){}

    port::Status OCLExecutor::SynchronousMemcpy(void* host_dst,
            const DeviceMemoryBase& gpu_src,
            uint64 size){}

    port::Status OCLExecutor::SynchronousMemcpyDeviceToDevice(DeviceMemoryBase* gpu_dst,
            const DeviceMemoryBase& gpu_src,
            uint64 size){}
}//namespace stream_executor
