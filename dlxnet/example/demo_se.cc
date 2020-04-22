#include "dlxnet/stream_executor/stream_executor.h"
#include "dlxnet/stream_executor/multi_platform_manager.h"

namespace se = stream_executor;

void opencl_main(){
    static constexpr const char *FILE_NAME="../dlxnet/example/cl/vec_add.ocl";

    // The number of arguments expected by the kernel described in
    // KERNEL_PTX_TEMPLATE.
    static constexpr int KERNEL_ARITY = 3;

    static constexpr int N = 10;
    static constexpr int bytes = N*sizeof(float);

    float* A = new float[N];
    float* B = new float[N];
    float* C = new float[N];
    // mock data
    for(int i=0;i<N;i++){
        A[i] = 1.0;
        B[i] = 10.0;
    }

    static constexpr const char* platform_name = "OpenCL";

    // The name of the kernel described in KERNEL_PTX.
    static constexpr const char *KERNEL_NAME = "vector_add";

    // Get a CUDA Platform object. (Other platforms such as OpenCL are also
    // supported.)
    se::Platform *platform =
        se::MultiPlatformManager::PlatformWithName(platform_name).ValueOrDie();

    // Get a StreamExecutor for the chosen Platform. Multiple devices are
    // supported, we indicate here that we want to run on device 0.
    const int device_ordinal = 0;
    se::StreamExecutor *executor =
        platform->ExecutorForDevice(device_ordinal).ValueOrDie();

    // Create a MultiKernelLoaderSpec, which knows where to find the code for our
    // kernel. In this case, the code is stored in memory as a PTX string.
    //
    // Note that the "arity" and name specified here must match  "arity" and name
    // of the kernel defined in the PTX string.
    se::MultiKernelLoaderSpec kernel_loader_spec(KERNEL_ARITY);
    // kernel_loader_spec.AddCudaPtxInMemory(KERNEL_PTX, KERNEL_NAME);
    kernel_loader_spec.AddOpenCLTextOnDisk(FILE_NAME, KERNEL_NAME);

    // Next create a kernel handle, which we will associate with our kernel code
    // (i.e., the PTX string).  The type of this handle is a bit verbose, so we
    // create an alias for it.
    //
    // This specific type represents a kernel that takes two arguments: a floating
    // point value and a pointer to a floating point value in device memory.
    //
    // A type like this is nice to have because it enables static type checking of
    // kernel arguments when we enqueue work on a stream.
    using KernelType = se::TypedKernel<se::DeviceMemory<float> *, se::DeviceMemory<float> *, se::DeviceMemory<float> *>;

    // Now instantiate an object of the specific kernel type we declared above.
    // The kernel object is not yet connected with the device code that we want it
    // to run (that happens with the call to GetKernel below), so it cannot be
    // used to execute work on the device yet.
    //
    // However, the kernel object is not completely empty when it is created. From
    // the StreamExecutor passed into its constructor it knows which platform it
    // is targeted for, and it also knows which device it will run on.
    KernelType kernel(executor);

    // Use the MultiKernelLoaderSpec defined above to load the kernel code onto
    // the device pointed to by the kernel object and to make that kernel object a
    // handle to the kernel code loaded on that device.
    //
    // The MultiKernelLoaderSpec may contain code for several different platforms,
    // but the kernel object has an associated platform, so there is no confusion
    // about which code should be loaded.
    //
    // After this call the kernel object can be used to launch its kernel on its
    // device.
    auto status = executor->GetKernel(kernel_loader_spec, &kernel);
    if(!status.ok()){
        LOG(FATAL)<<status;
    }

    // Allocate memory in the device memory space to hold the result of the kernel
    // call. This memory will be freed when this object goes out of scope.
    se::ScopedDeviceMemory<float> input0 = executor->AllocateOwnedArray<float>(N);
    se::ScopedDeviceMemory<float> input1 = executor->AllocateOwnedArray<float>(N);
    se::ScopedDeviceMemory<float> output = executor->AllocateOwnedArray<float>(N);

    // upload data
    executor->SynchronousMemcpyH2D(A, bytes, input0.ptr());
    executor->SynchronousMemcpyH2D(B, bytes, input1.ptr());

    // Create a stream on which to schedule device operations.
    se::Stream stream(executor);

    // Schedule a kernel launch on the new stream and block until the kernel
    // completes. The kernel call executes asynchronously on the device, so we
    // could do more work on the host before calling BlockHostUntilDone.
    // const float kernel_input_argument = 42.5f;
    stream.Init()
        .ThenLaunch(se::ThreadDim(), se::BlockDim(N), kernel,
                input0.ptr(), input1.ptr(), output.ptr())
        .BlockHostUntilDone();

    // Copy the result of the kernel call from device back to the host.
    // executor->SynchronousMemcpyD2H(output.cref(), bytes, C);

    // Verify that the correct result was computed.
    // assert((kernel_input_argument + MYSTERY_VALUE) == host_result);
    executor->SynchronousMemcpyD2H(output.cref(), bytes, C);
    for(int i=0; i<N; ++i){
        std::cout<<C[i]<<std::endl;
    }
}


int main(){
    opencl_main();
    return 0;
}
