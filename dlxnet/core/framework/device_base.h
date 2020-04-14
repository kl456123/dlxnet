#ifndef DLXNET_CORE_FRAMEWORK_DEVICE_BASE_H_
#define DLXNET_CORE_FRAMEWORK_DEVICE_BASE_H_
#include "dlxnet/core/platform/refcount.h"
#include "dlxnet/core/lib/errors.h"
#include "dlxnet/core/lib/status.h"
#include "dlxnet/core/lib/stringpiece.h"
#include "dlxnet/core/framework/tensor.h"
#include "dlxnet/core/framework/allocator.h"

namespace Eigen {
    struct ThreadPoolDevice;
#ifdef DLXNET_USE_SYCL
    struct SyclDevice;
#endif
}  // end namespace Eigen

namespace dlxnet{
    class Device;
    class DeviceAttributes;
    class Env;
    namespace thread {
        class ThreadPool;
    }

    // A class that devices can subclass to pass around
    // Device-specific context to OpKernels.
    class DeviceContext : public core::RefCounted {
        public:
            ~DeviceContext() override {}
            // "cpu_tensor" is a tensor on a CPU. Copies "cpu_tensor" into
            // "device_tensor" which is on a non-CPU device "device". "device_tensor"
            // must be allocated to be of the same size as "cpu_tensor".
            virtual void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                    Tensor* device_tensor, StatusCallback done,
                    bool sync_dst_compute = true) const {
                done(errors::Internal("Unrecognized device type in CPU-to-device Copy"));
            }
            // Copies a tensor in this device.
            virtual void CopyTensorInSameDevice(const Tensor* input_tensor,
                    Device* device, Tensor* output_tensor,
                    StatusCallback done) const {
                done(errors::Unimplemented("Copy in same device not implemented."));
            }
            // "device_tensor" is a tensor on a non-CPU device.  Copies
            // device_tensor into "cpu_tensor".  "cpu_tensor" must be allocated
            // to be of the same size as "device_tensor".
            virtual void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                    StringPiece tensor_name, Device* device,
                    Tensor* cpu_tensor, StatusCallback done) {
                done(errors::Internal("Unrecognized device type in device-to-CPU Copy"));
            }
    };

    class DeviceBase{
        public:
            explicit DeviceBase(Env* env):env_(env){}
            virtual ~DeviceBase();
            Env* env()const {return env_;}
            // Return the Allocator implementation to use based on the allocator
            // attributes requested.  See allocator.h for more details.
            virtual Allocator* GetAllocator(AllocatorAttributes /*attr*/) {
                LOG(FATAL) << "GetAllocator() is not implemented.";
                return nullptr;
            }
            // Unimplemented by default
            virtual const DeviceAttributes& attributes() const;
            virtual const string& name() const;
            virtual DeviceBase* UnderlyingDevice() { return this; }
            virtual const DeviceBase* UnderlyingDevice() const { return this; }
            // The preferred thread pool for this device. If it is nullptr, the system
            // automatically assigns a thread pool for execution.
            virtual thread::ThreadPool* tensorflow_device_thread_pool() {
                return device_thread_pool_;
            }
            // Copies `input_tensor` to `output_tensor`, where both tensors are on this
            // device. This function assumes that `output_tensor` has already been
            // allocated with a buffer that is large enough to hold `input_tensor`'s data.
            // Calls `done` from a device-specific thread after copy is finished, which
            // may be the same as calling thread.
            //
            // NOTE(ayushd): This function is for TensorFlow internal use only.  Deep copy
            // is discouraged and should not be used in OpKernels.
            virtual void CopyTensorInSameDevice(const Tensor* input_tensor,
                    Tensor* output_tensor,
                    const DeviceContext* device_context,
                    StatusCallback done) {
                done(errors::Internal("Device ", name(), " does not implement ",
                            "CopyTensorInSameDevice"));
            }

            // Materializes the given TensorProto into 'tensor' stored in Device
            // memory.  Most devices will want to override this.
            //
            // TODO(vrv): We should be able to put this function into
            // OpKernelContext and handle the copies from device memory via send
            // and receive nodes, instead of requiring that each device handle
            // the copies here as well as in copy ops.
            virtual Status MakeTensorFromProto(const TensorProto& tensor_proto,
                    const AllocatorAttributes alloc_attrs,
                    Tensor* tensor) {
                return errors::Internal("Device does not implement MakeTensorFromProto()");
            }

            // eigen utils
            void set_eigen_cpu_device(Eigen::ThreadPoolDevice* d);
            const Eigen::ThreadPoolDevice* eigen_cpu_device();
            void set_eigen_cpu_thread_pool(thread::ThreadPool* t){
                cpu_thread_pool_ = t;
            }

            bool has_eigen_cpu_device()const{
                return !eigen_cpu_devices_.empty();
            }

            const thread::ThreadPool* eigen_cpu_thread_pool()const{
                CHECK(cpu_thread_pool_!=nullptr);
                return cpu_thread_pool_;
            }
        protected:
            // Does not take ownership.
            void set_device_thread_pool(thread::ThreadPool* thread_pool) {
                device_thread_pool_ = thread_pool;
            }

        private:
            Env* const env_;
            thread::ThreadPool* device_thread_pool_ = nullptr;
            thread::ThreadPool* cpu_thread_pool_ = nullptr;
            std::vector<Eigen::ThreadPoolDevice*> eigen_cpu_devices_;
    };
}// namespace dlxnet

#endif
