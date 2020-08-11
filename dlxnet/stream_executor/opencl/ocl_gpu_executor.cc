#include "dlxnet/stream_executor/opencl/ocl_gpu_executor.h"
#include "dlxnet/stream_executor/opencl/ocl_driver.h"
#include "dlxnet/stream_executor/opencl/ocl_event.h"
#include "dlxnet/core/lib/core/errors.h"
#include "absl/strings/str_format.h"

namespace stream_executor{
    namespace{
        static GpuEvent* AsGpuEvent(Event* event) {
            DCHECK(event != nullptr);
            return static_cast<GpuEvent*>(event->implementation());
        }
    } // namespace

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

    port::Status OCLExecutor::GetKernel(const MultiKernelLoaderSpec &spec,
            KernelBase *kernel){
        OCLKernel* ocl_kernel = AsGpuKernel(kernel);
        cl::Program program;
        const string *kernelname;
        VLOG(3) << "GetKernel on kernel " << kernel << " : " << kernel->name();

        // cache them after program loaded
        if(spec.has_ocl_text_on_disk()){
            kernelname = &spec.ocl_text_on_disk().kernelname();
            SE_RETURN_IF_ERROR(LoadProgramFromText(spec.ocl_text_on_disk().filename(), &program));
        }else if(spec.has_ocl_binary_on_disk()){
            kernelname = &spec.ocl_binary_on_disk().kernelname();
            SE_RETURN_IF_ERROR(LoadProgramFromBin(spec.ocl_binary_on_disk().filename(), &program));
        }else if(spec.has_ocl_text_in_memory()){
            return port::InternalError(
                    "Loading OpenCL kernel from memory is not supported");
        }else{
            return port::InternalError("No method of loading OpenCL kernel provided");
        }

        VLOG(2) << "getting kernel " << *kernelname << " from program " << &program;
        if (!OCLDriver::GetProgramKernel(context_, program, kernelname->c_str(),
                    ocl_kernel->gpu_function_ptr())) {
            return port::InternalError("Could not find the corresponding function");
        }

        // We have to trust the kernel loader spec arity because there doesn't appear
        // to be a way to reflect on the number of expected arguments w/the CUDA API.
        ocl_kernel->set_arity(spec.arity());

        KernelMetadata kernel_metadata;
        SE_RETURN_IF_ERROR(GetKernelMetadata(ocl_kernel, &kernel_metadata));
        kernel->set_metadata(kernel_metadata);
        kernel->set_name(*kernelname);
        return port::Status::OK();
    }

    port::Status OCLExecutor::LoadProgramFromBin(absl::string_view fname,
            GpuModuleHandle* module){
        return Status::OK();
    }

    port::Status OCLExecutor::GetKernelMetadata(OCLKernel* ocl_kernel,
            KernelMetadata* kernel_metadata) {
        return port::Status::OK();
    }




    port::Status OCLExecutor::LoadProgramFromText(absl::string_view fname,
            GpuModuleHandle* out){
        // check it from cache first
        GpuModuleHandle module;
        if(module()==nullptr){
            SE_RETURN_IF_ERROR(OCLDriver::LoadText(context_, fname, &module));
        }else{
            VLOG(3) << "fname: " << fname
                << " is already loaded as module " << module();
        }
        *out = module;
        // cache here then
        return Status::OK();
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
        CHECK(context_());
        return DeviceMemoryBase(OCLDriver::DeviceAllocate(context_, size), size);
    }

    void OCLExecutor::Deallocate(DeviceMemoryBase *mem){
        OCLDriver::DeviceDeallocate(context_, AsOCLDevicePtr(mem));
    }

    // implementation
    std::unique_ptr<internal::TimerInterface> OCLExecutor::GetTimerImplementation(){
    }

    std::unique_ptr<internal::StreamInterface> OCLExecutor::GetStreamImplementation(){
        return std::unique_ptr<internal::StreamInterface>(new GpuStream(this));
    }

    std::unique_ptr<internal::KernelInterface> OCLExecutor::CreateKernelImplementation(){
        return std::unique_ptr<internal::KernelInterface>(new OCLKernel);
    }

    std::unique_ptr<internal::EventInterface> OCLExecutor::CreateEventImplementation(){
    }

    // event
    port::Status OCLExecutor::WaitForEvent(Stream* stream, Event* event){
        if (OCLDriver::WaitStreamOnEvent(context_, AsGpuStream(stream)->gpu_stream(),
                    AsGpuEvent(event)->gpu_event())) {
            return port::Status::OK();
        } else {
            return port::Status(
                    port::error::INTERNAL,
                    absl::StrFormat("error recording waiting for CUDA event on stream %p",
                        stream));
        }
    }

    port::Status OCLExecutor::RecordEvent(Stream* stream, Event* event){
        return AsGpuEvent(event)->Record(AsGpuStream(stream));
    }

    port::Status OCLExecutor::AllocateEvent(Event* event){
        return port::Status::OK();
    }

    port::Status OCLExecutor::DeallocateEvent(Event* event){
        return port::Status::OK();
    }

    Event::Status OCLExecutor::PollForEventStatus(Event* event) {
        return Event::Status::kUnknown;
    }

    port::Status OCLExecutor::BlockHostUntilDone(Stream* stream){
        return port::Status::OK();
    }

    // stream
    bool OCLExecutor::AllocateStream(Stream* stream){
        return AsGpuStream(stream)->Init();
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
            const void* host_src, uint64 size){
        return OCLDriver::SynchronousMemcpyH2D(context_, AsOCLDevicePtr(gpu_dst),
                host_src, size);
    }

    port::Status OCLExecutor::SynchronousMemcpy(void* host_dst,
            const DeviceMemoryBase& gpu_src, uint64 size){
        return OCLDriver::SynchronousMemcpyD2H(context_, host_dst,
                AsOCLDevicePtr(gpu_src), size);

    }

    port::Status OCLExecutor::SynchronousMemcpyDeviceToDevice(DeviceMemoryBase* gpu_dst,
            const DeviceMemoryBase& gpu_src,
            uint64 size){
        return OCLDriver::SynchronousMemcpyD2D(context_, AsOCLDevicePtr(gpu_dst),
                AsOCLDevicePtr(gpu_src), size);
    }

    bool OCLExecutor::Memcpy(Stream* stream, void* host_dst,
            const DeviceMemoryBase& gpu_src, uint64 size) {
        return OCLDriver::AsynchronousMemcpyD2H(context_, host_dst,
                AsOCLDevicePtr(gpu_src), size,
                AsGpuStreamValue(stream));
    }

    bool OCLExecutor::Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst,
            const void* host_src, uint64 size) {
        return OCLDriver::AsynchronousMemcpyH2D(context_, AsOCLDevicePtr(gpu_dst),
                host_src, size,
                AsGpuStreamValue(stream));
    }

    bool OCLExecutor::MemcpyDeviceToDevice(Stream* stream,
            DeviceMemoryBase* gpu_dst,
            const DeviceMemoryBase& gpu_src,
            uint64 size) {
        return OCLDriver::AsynchronousMemcpyD2D(context_, AsOCLDevicePtr(gpu_dst),
                AsOCLDevicePtr(gpu_src), size,
                AsGpuStreamValue(stream));
    }

    port::Status OCLExecutor::Launch(Stream *stream, const ThreadDim &thread_dims,
            const BlockDim &block_dims, const KernelBase &kernel,
            const KernelArgsArrayBase &args){
        CHECK_EQ(kernel.Arity(), args.number_of_arguments());
        GpuStreamHandle ocl_stream = AsGpuStreamValue(stream);
        const OCLKernel* ocl_kernel = AsGpuKernel(&kernel);
        GpuFunctionHandle ocl_func = ocl_kernel->AsGpuFunctionHandle();

        if (ocl_kernel->GetPreferredCacheConfig() !=
                KernelCacheConfig::kNoPreference) {
        }

        // set args here
        // loop using iterator
        KernelArgIterator iterator = args.arg_iterator();
        size_t arg_index = 0;
        while(iterator.has_next()){
            KernelArg kernel_arg = iterator.next();
            CHECK(kernel_arg.address);
            CHECK(!kernel_arg.is_shared);
            try{
                ocl_func.setArg(arg_index, kernel_arg.size, kernel_arg.address);
            }catch(cl::Error error){
                return ::dlxnet::errors::Internal("arg_index: ",arg_index," ",
                        error.what(), "(", error.err(), ")");
            }
            ++arg_index;
        }

        cl::NDRange gws = {block_dims.x, block_dims.y, block_dims.z};
        cl::NDRange lws = {thread_dims.x, thread_dims.y, thread_dims.z};
        return OCLDriver::LaunchKernel(context_, ocl_func, gws, lws, ocl_stream);
    }
}//namespace stream_executor
