#ifndef DLXNET_CORE_COMMON_RUNTIME_DEVICE_H_
#define DLXNET_CORE_COMMON_RUNTIME_DEVICE_H_
#include "dlxnet/core/framework/device_attributes.pb.h"
#include "dlxnet/core/framework/device_base.h"
#include "dlxnet/core/framework/types.h"
#include "dlxnet/core/framework/op_kernel.h"
#include "dlxnet/core/lib/core/status.h"
#include "dlxnet/core/platform/macros.h"
#include "dlxnet/core/util/device_name_utils.h"
#include "dlxnet/core/graph/types.h"


namespace dlxnet{
    class Device : public DeviceBase{
        public:
            // Callback type that takes a Status and returns void.
            typedef std::function<void(const Status&)> DoneCallback;
            Device(Env* env, const DeviceAttributes& device_attributes);
            ~Device() override;
            const string& device_type() const { return device_attributes_.device_type(); }
            // Returns an aggregation of device attributes.
            const DeviceAttributes& attributes() const override {
                return device_attributes_;
            }

            // Full name of this device (see top comment).
            const string& name() const override { return device_attributes_.name(); }

            // Parsed name of this device
            const DeviceNameUtils::ParsedName& parsed_name() const {
                return parsed_name_;
            }
            // Performs the actual compute function.
            //
            // Subclasses may override this function if they wish to perform
            // some initialization before each compute.
            virtual void Compute(OpKernel* op_kernel, OpKernelContext* context) {
                op_kernel->Compute(context);
            }

            // Asynchronous kernel's compute.
            virtual void ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                    AsyncOpKernel::DoneCallback done) {
                op_kernel->ComputeAsync(context, std::move(done));
            }
            // Blocks until all operations queued on the device at the time of
            // the call have completed.  Returns any error pending on the device
            // at completion.
            virtual Status Sync() = 0;

            // Calls the given callback when all operations queued on the device at the
            // time of the call have completed. The callback is passed any error pending
            // on the device at completion.
            // TODO(b/112409994): Consolidate these two APIs, removing the synchronous
            // version.
            virtual void Sync(const DoneCallback& done);
            // Summarizes the status of this Device, for debugging.
            string DebugString() const { return device_attributes_.DebugString(); }
            // Assembles the parameter components into a complete DeviceAttributes value.
            static DeviceAttributes BuildDeviceAttributes(
                    const string& name, DeviceType device, Bytes memory_limit,
                    const DeviceLocality& locality, const string& physical_device_desc);

            static DeviceAttributes BuildDeviceAttributes(
                    const string& name, DeviceType device, Bytes memory_limit,
                    const DeviceLocality& locality) {
                // Pass in an empty string as physical device name.
                return BuildDeviceAttributes(name, device, memory_limit, locality, "");
            }

            // Sets `out_context` a new DeviceContext* for executing a graph, or nullptr
            // if the device does not support contexts. Returns an error status if any
            // error occurred while trying to create a context, otherwise OK.
            //
            // The caller takes ownership of one reference on the output DeviceContext*,
            // and should call Unref().
            virtual Status TryGetDeviceContext(DeviceContext** out_context) {
                *out_context = nullptr;
                return Status::OK();
            }
            virtual bool IsLocal() const { return true; }
        private:
            DeviceNameUtils::ParsedName parsed_name_;
            const DeviceAttributes device_attributes_;
            TF_DISALLOW_COPY_AND_ASSIGN(Device);
    };
}


#endif
