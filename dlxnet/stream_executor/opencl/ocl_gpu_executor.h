#ifndef DLXNET_STREAM_EXECUTOR_OCL_OCL_GPU_EXECUTOR_H_
#define DLXNET_STREAM_EXECUTOR_OCL_OCL_GPU_EXECUTOR_H_
#include <set>
#include <unordered_map>

#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "dlxnet/stream_executor/event.h"
#include "dlxnet/stream_executor/lib/status.h"
#include "dlxnet/stream_executor/lib/statusor.h"
#include "dlxnet/stream_executor/platform.h"
#include "dlxnet/stream_executor/platform/port.h"
#include "dlxnet/stream_executor/platform/thread_annotations.h"
#include "dlxnet/stream_executor/stream_executor_internal.h"
#include "dlxnet/stream_executor/gpu/gpu_types.h"
#include "dlxnet/stream_executor/opencl/ocl_driver.h"
#include "dlxnet/stream_executor/opencl/ocl_kernel.h"
#include "dlxnet/stream_executor/kernel.h"

namespace stream_executor{
    class OCLExecutor: public internal::StreamExecutorInterface{
        public:
            // sub_platform indicates the subplatform used in this executor; it must
            // be a CUDA type.
            explicit OCLExecutor(const PluginConfig& plugin_config)
                : device_(0),
                context_(nullptr),
                device_ordinal_(0),
                version_(0),
                plugin_config_(plugin_config) {}
            ~OCLExecutor()override;

            port::Status Init(int device_ordinal, DeviceOptions device_options) override;

            port::Status GetKernel(const MultiKernelLoaderSpec &spec,
                    KernelBase *kernel) override;
            port::Status Launch(Stream *stream, const ThreadDim &thread_dims,
                    const BlockDim &block_dims, const KernelBase &kernel,
                    const KernelArgsArrayBase &args) override ;

            // Collects metadata for the specified kernel.
            port::Status GetKernelMetadata(OCLKernel* cuda_kernel,
                    KernelMetadata* kernel_metadata);

            DeviceMemoryBase Allocate(uint64 size, int64 memory_space) override;

            // copy function device to host, host to device and device to device
            port::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                    const void* host_src, uint64 size) override;

            port::Status SynchronousMemcpy(void* host_dst,
                    const DeviceMemoryBase& gpu_src,
                    uint64 size) override;

            port::Status SynchronousMemcpyDeviceToDevice(DeviceMemoryBase* gpu_dst,
                    const DeviceMemoryBase& gpu_src,
                    uint64 size) override;

            port::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
                const override {
                    return CreateDeviceDescription(device_ordinal_);
                }

            static port::StatusOr<std::unique_ptr<DeviceDescription>>
                CreateDeviceDescription(int device_ordinal);

            // Given const GPU memory, returns a libcuda device pointer datatype, suitable
            // for passing directly to libcuda APIs.
            //
            // N.B. we must lose constness in order to pass a suitable type to the existing
            // libcuda APIs, so the caller should take care to only pass the result of const
            // GPU memory conversions to libcuda functions which will honor constness.
            static GpuDevicePtr AsOCLDevicePtr(const DeviceMemoryBase &gpu_mem) {
                return reinterpret_cast<GpuDevicePtr>(const_cast<void*>(gpu_mem.opaque()));
            }

            // See description on const version above.
            static GpuDevicePtr AsOCLDevicePtr(DeviceMemoryBase *gpu_mem) {
                return AsOCLDevicePtr(*gpu_mem);
            }

            // some other implementation managed by stream executor,
            // like stream, timer and event
            // (TODO use Stream or StreamInterface in argument list)
            bool AllocateStream(Stream* stream) override;

            void DeallocateStream(Stream* stream) override;

            bool AllocateTimer(Timer* timer) override;

            void DeallocateTimer(Timer* timer) override;

            bool StartTimer(Stream* stream, Timer* timer) override;

            bool StopTimer(Stream* stream, Timer* timer) override;

            port::Status AllocateEvent(Event* event) override;

            port::Status DeallocateEvent(Event* event) override;

            port::Status RecordEvent(Stream* stream, Event* event) override;

            port::Status WaitForEvent(Stream* stream, Event* event) override;

            Event::Status PollForEventStatus(Event* event) override;

            port::Status BlockHostUntilDone(Stream* stream) override;

            int PlatformDeviceCount() override { return OCLDriver::GetDeviceCount(); }

            void Deallocate(DeviceMemoryBase* mem) override;

            std::unique_ptr<internal::EventInterface> CreateEventImplementation()
                override;

            std::unique_ptr<internal::KernelInterface> CreateKernelImplementation()
                override;

            std::unique_ptr<internal::StreamInterface> GetStreamImplementation() override;

            std::unique_ptr<internal::TimerInterface> GetTimerImplementation() override;

            GpuContext gpu_context()const{
                return context_;
            }

            // (supported on OpenCL only)
            port::Status LoadProgramFromText(absl::string_view fname, GpuModuleHandle* module)
                EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);

            // (supported on OpenCL only)
            port::Status LoadProgramFromBin(absl::string_view fname, GpuModuleHandle* module)
                EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);

        private:

            // Guards the in-memory-module mapping.
            absl::Mutex in_memory_modules_mu_;

            std::map<const char*, GpuModuleHandle> in_memory_modules_
                GUARDED_BY(in_memory_modules_mu_);

            // The device ordinal value that this executor was initialized with; recorded
            // for use in getting device metadata. Immutable post-initialization.
            int device_ordinal_;

            cl::Device device_;

            cl::Context context_;

            // GPU ISA version for device_.
            int version_;

            // The plugin configuration associated with this instance.
            PluginConfig plugin_config_;

            std::unordered_map<string, GpuFunctionHandle> programs_map_;

            SE_DISALLOW_COPY_AND_ASSIGN(OCLExecutor);


    };
}//namespace stream_executor

#endif
