#include "dlxnet/core/common_runtime/gpu/gpu_device.h"
#include "dlxnet/core/common_runtime/gpu/gpu_id_manager.h"
#include "dlxnet/core/common_runtime/gpu/gpu_id_utils.h"
#include "dlxnet/core/common_runtime/gpu/gpu_util.h"
#include "dlxnet/core/common_runtime/gpu/gpu_process_state.h"
#include "dlxnet/core/platform/types.h"
#include "dlxnet/core/lib/core/status.h"
#include "dlxnet/core/lib/strings/str_util.h"
#include "dlxnet/core/lib/core/notification.h"
#include "dlxnet/core/util/env_var.h"
#include "dlxnet/core/platform/logging.h"



namespace dlxnet{

    // This factory helps to ensure that different GPU device objects that refer to
    // the same physical device and stream group id use the same stream group
    // object (and therefore the same CUDA streams). This is necessary since there
    // is a single memory allocator per device (see ProcessState::GetGPUAllocator)
    // and allocators must not be shared across streams.
    class BaseGPUDevice::StreamGroupFactory {
        public:
            // Returns the unique stream group for use with the stream defined by
            // {tf_gpu_id, stream_group_within_gpu}, creating it if it does not yet
            // exist.
            // This function is thread safe.
            BaseGPUDevice::StreamGroup* GetOrCreate(TfGpuId tf_gpu_id,
                    int stream_group_within_gpu,
                    se::StreamExecutor* executor,
                    const GPUOptions& options) {
                mutex_lock guard(lock_);
                StreamGroup* group =
                    &streams_[key_type(tf_gpu_id, stream_group_within_gpu)];
                if(!group->compute){
                    group->compute = new se::Stream(executor);
                    group->compute->Init();
                    VLOG(2) << "Created stream[" << stream_group_within_gpu
                        << "] = " << group->compute;

                    group->host_to_device = new se::Stream(executor);
                    group->host_to_device->Init();
                    VLOG(2) << "Created host_to_device_stream[" << stream_group_within_gpu
                        << "] = " << group->host_to_device;

                    group->device_to_host = new se::Stream(executor);
                    group->device_to_host->Init();
                    VLOG(2) << "Created device_to_host_stream[" << stream_group_within_gpu
                        << "] = " << group->device_to_host;

                    int num_d2d_streams =
                        options.experimental().num_dev_to_dev_copy_streams();
                    if (num_d2d_streams == 0) num_d2d_streams = 1;
                    if (num_d2d_streams < 1 || num_d2d_streams > 4) {
                        LOG(ERROR)
                            << "Illegal GPUOptions.experimental.num_dev_to_dev_copy_streams="
                            << num_d2d_streams << " set to 1 instead.";
                        num_d2d_streams = 1;
                    }

                    for (int i = 0; i < num_d2d_streams; ++i) {
                        se::Stream* stream = new se::Stream(executor);
                        stream->Init();
                        group->device_to_device.push_back(stream);
                        VLOG(2) << "Created device_to_device_stream[" << stream_group_within_gpu
                            << "] = " << group->device_to_device.back();
                    }
                }
                return group;
            }

            // Returns a reference to the StreamGroupFactory singleton. Note that this is
            // never destroyed, so the objects it owns are never deleted.
            static StreamGroupFactory& Global() {
                static StreamGroupFactory* instance = new StreamGroupFactory();
                return *instance;
            }

        private:
            mutex lock_;
            using key_type = std::tuple<int, int>;
            std::map<key_type, StreamGroup> streams_;

            // StreamGroupFactory cannot be created directly; Call
            // StreamGroupFactory::Global() to get the global instance.
            StreamGroupFactory() = default;
            TF_DISALLOW_COPY_AND_ASSIGN(StreamGroupFactory);
    };

    BaseGPUDevice::BaseGPUDevice(const SessionOptions& options, const string& name,
            Bytes memory_limit, const DeviceLocality& locality,
            TfGpuId tf_gpu_id,
            const string& physical_device_desc,
            Allocator* gpu_allocator, Allocator* cpu_allocator,
            bool sync_every_op)
        : LocalDevice(options, Device::BuildDeviceAttributes(name, DEVICE_GPU,
                    memory_limit, locality,
                    physical_device_desc)),
        gpu_allocator_(gpu_allocator),
        cpu_allocator_(cpu_allocator),
        tf_gpu_id_(tf_gpu_id),
        sync_every_op_(sync_every_op) {
            // GPUProcessState::singleton()->EnableGPUDevice();
        }

    BaseGPUDevice::~BaseGPUDevice() {
        // delete gpu_device_info_;
        // gpu_allocator_->DeallocateRaw(scratch_);
        device_context_->Unref();
    }

    string BaseGPUDevice::ComputeOpKernelDebugString(const OpKernel& op_kernel,
            const int& stream_id) {
        return strings::StrCat(op_kernel.name(), " op ", op_kernel.type_string(),
                " on GPU ", tf_gpu_id_, " stream[", stream_id,
                "]");
    }

    Status BaseGPUDevice::Init(const SessionOptions& options) {
        auto executor_status = GpuIdUtil::ExecutorForTfGpuId(tf_gpu_id_);
        if (!executor_status.status().ok()) {
            return errors::Internal("Failed to get StreamExecutor for device ",
                    tf_gpu_id_);
        }

        executor_ = executor_status.ValueOrDie();

        stream_ = StreamGroupFactory::Global().GetOrCreate(
                tf_gpu_id_, 0, executor_, options.config.gpu_options());

        device_context_ =
            new GPUDeviceContext(0, stream_->compute,
                    stream_->host_to_device, stream_->device_to_host,
                    stream_->device_to_device);
        timestamped_allocator_ =
            options.config.gpu_options().experimental().timestamped_allocator();

        device_info_ = new DeviceInfo;
        device_info_->stream = stream_->compute;
        device_info_->default_context = device_context_;
        // device_info_->event_mgr = em_;
        PlatformGpuId platform_gpu_id;
        TF_RETURN_IF_ERROR(
                GpuIdManager::TfToPlatformGpuId(tf_gpu_id_, &platform_gpu_id));
        device_info_->gpu_id = platform_gpu_id;
        set_device_info(device_info_);

        // Whether and how the GPU device uses its own threadpool.
        // This option is experimental. Once we confirm the best setting, we
        // may change the default behavior and completely remove this flag.
        // Default values might change in future releases.
        // Possible values:
        //   * global: GPU uses threads shared with CPU in the main compute
        //          thread-pool. This is currently the default.
        //   * gpu_private: GPU uses threads dedicated to this device.
        //   * gpu_shared: All GPUs share a dedicated thread pool.
        string gpu_thread_mode;
        TF_RETURN_IF_ERROR(
                ReadStringFromEnvVar("TF_GPU_THREAD_MODE", "global", &gpu_thread_mode));
        gpu_thread_mode = absl::AsciiStrToLower(gpu_thread_mode);
        if (gpu_thread_mode != "global") {
            int64 gpu_thread_count = -1;
            // Default to two threads. One for device compute and another for memory
            // copies.
            TF_RETURN_IF_ERROR(
                    ReadInt64FromEnvVar("TF_GPU_THREAD_COUNT", 2, &gpu_thread_count));
            if (gpu_thread_mode == "gpu_private") {
                // TODO(zhengxq): since these threads only serve a single GPU device,
                //   we should set the device context once for each thread, and avoid
                //   setting them for each kernel.
                // TODO(zhengxq): pin the thread to the same socket of the target GPU.
                thread_pool_.reset(new thread::ThreadPool(
                            options.env, ThreadOptions(),
                            strings::StrCat("gpu_private_", tf_gpu_id_),
                            static_cast<int32>(gpu_thread_count),
                            !options.config.experimental().disable_thread_spinning(),
                            /*allocator=*/nullptr));
            }else if (gpu_thread_mode == "gpu_shared") {
            }else{
            }
        }

        return Status::OK();
    }

    void BaseGPUDevice::CopyTensorInSameDevice(const Tensor* input_tensor,
            Tensor* output_tensor,
            const DeviceContext* device_context,
            StatusCallback done) {
        // GPUUtil::CopyGPUTensorToSameGPU(static_cast<Device*>(this), device_context,
        // input_tensor, output_tensor, std::move(done));
    }

    // Based on the semantics of Device::Sync this call should wait for
    // all streams not just the current one.
    Status BaseGPUDevice::Sync() { return GPUUtil::SyncAll(this); }

    void BaseGPUDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
        // NOTE(tucker): We need to discriminate between Eigen GPU
        // operations and all others.  If an operation is Eigen
        // implemented (or otherwise tries to launch a GPU kernel
        // directly), we need to establish a stacked-scoped environment
        // that directs it to execute on the proper device.  Otherwise we
        // expect the Op to use StreamExecutor directly and correctly.
        GPUDeviceContext* gpu_device_context = device_context_;
        if (context->op_device_context() != nullptr) {
            gpu_device_context =
                static_cast<GPUDeviceContext*>(context->op_device_context());
        }
        const bool vlog_1 = VLOG_IS_ON(1);
        const auto stream_id = gpu_device_context->stream_id();
        if (vlog_1) {
            VLOG(1) << "GpuDevice::ComputeHelper "
                << ComputeOpKernelDebugString(*op_kernel, stream_id);
        }

        op_kernel->Compute(context);
        if (context->status().ok()) {
            if(vlog_1){
                VLOG(1) << "GpuDevice::ComputeHelper scheduled "
                    << ComputeOpKernelDebugString(*op_kernel, stream_id);
            }
        }else{
            if (vlog_1) {
                VLOG(1) << "GpuDevice::ComputeHelper failed to schedule "
                    << ComputeOpKernelDebugString(*op_kernel, stream_id);
            }
        }
    }
    void BaseGPUDevice::ComputeAsync(AsyncOpKernel* op_kernel,
            OpKernelContext* context,
            AsyncOpKernel::DoneCallback done) {
        GPUDeviceContext* gpu_device_context = device_context_;
        if (context->op_device_context() != nullptr) {
            gpu_device_context =
                static_cast<GPUDeviceContext*>(context->op_device_context());
        }
        // se::Stream* stream = gpu_device_context->stream();
        const auto stream_id = gpu_device_context->stream_id();

        VLOG(1) << "GpuDevice::ComputeAsync " << op_kernel->name() << " op "
            << op_kernel->type_string() << " on GPU" << tf_gpu_id_ << " stream["
            << stream_id << "]";

        op_kernel->ComputeAsync(context, std::move(done));
    }

    Status BaseGPUDevice::MaybeCopyTensorToGPU(
            const AllocatorAttributes& alloc_attrs, const Tensor& from, Tensor* to,
            StatusCallback done){
        if (alloc_attrs.on_host()) {
            *to = from;
            done(Status::OK());
            return Status::OK();
        } else {
            AllocationAttributes allocation_attr;
            auto* copy = new Tensor(GetAllocator(alloc_attrs), from.dtype(),
                    from.shape(), allocation_attr);
            // If the tensor is not initialized, we likely ran out of memory.
            if (!copy->IsInitialized()) {
                delete copy;
                Status err = errors::ResourceExhausted(
                        "OOM when allocating tensor of shape ", from.shape().DebugString(),
                        " and type ", DataTypeString(from.dtype()));
                done(err);
                return err;
            }
            auto wrapped_done = [to, copy, done = std::move(done)](const Status& s) {
                if (s.ok()) {
                    *to = std::move(*copy);
                }
                delete copy;
                done(s);
            };

            device_context_->CopyCPUTensorToDevice(
                    &from, this, copy, std::move(wrapped_done),
                    !timestamped_allocator_ /*sync_dst_compute*/);
            return Status::OK();
        }
    }

    Status BaseGPUDevice::MakeTensorFromProto(const TensorProto& tensor_proto,
            const AllocatorAttributes alloc_attrs,
            Tensor* tensor) {
        // first from disk to cpu memory
        AllocatorAttributes attr;
        attr.set_on_host(true);
        attr.set_gpu_compatible(true);
        Allocator* host_alloc = GetAllocator(attr);
        Tensor parsed(tensor_proto.dtype());
        if (!parsed.FromProto(host_alloc, tensor_proto)) {
            return errors::InvalidArgument("Cannot parse tensor from proto: ",
                    tensor_proto.DebugString());
        }
        if (parsed.dtype() == DT_VARIANT) {
        }else{
            Notification n;
            Status status;
            TF_RETURN_IF_ERROR(MaybeCopyTensorToGPU(alloc_attrs, parsed, tensor,
                        [&n, &status](const Status& s) {
                        status = s;
                        n.Notify();
                        }));
            n.WaitForNotification();
            return status;
        }
    }



    Status BaseGPUDeviceFactory::ListPhysicalDevices(std::vector<string>* devices) {

        // get gpu device count from platform manager first
        int device_count = 1;

        std::vector<PlatformGpuId> visible_gpu_order(device_count);
        int deviceNo = 0;
        std::generate(visible_gpu_order.begin(), visible_gpu_order.end(),
                [&deviceNo] { return deviceNo++; });

        std::vector<PlatformGpuId> valid_platform_gpu_ids;
        TF_RETURN_IF_ERROR(
                GetValidDeviceIds(visible_gpu_order, &valid_platform_gpu_ids));

        for (PlatformGpuId platform_gpu_id : valid_platform_gpu_ids) {
            const string device_name =
                strings::StrCat("/physical_device:GPU:", platform_gpu_id);
            devices->push_back(device_name);
        }
        return Status::OK();
    }

    namespace{
        // Parse 'visible_device_list' into a list of platform GPU ids.
        Status ParseVisibleDeviceList(const string& visible_device_list,
                std::vector<PlatformGpuId>* visible_gpu_order) {
            visible_gpu_order->clear();
            se::Platform* gpu_manager = GPUMachineManager();

            // If the user wants to remap the visible to virtual GPU mapping,
            // check for that here.
            if (visible_device_list.empty()) {
                visible_gpu_order->resize(gpu_manager->VisibleDeviceCount());
                // By default, visible to virtual mapping is unchanged.
                int deviceNo = 0;
                std::generate(visible_gpu_order->begin(), visible_gpu_order->end(),
                        [&deviceNo] { return deviceNo++; });
            }else{
                const std::vector<string> order_str =
                    str_util::Split(visible_device_list, ',');
                for (const string& platform_gpu_id_str : order_str) {
                    int32 platform_gpu_id;
                    if (!strings::safe_strto32(platform_gpu_id_str, &platform_gpu_id)) {
                        return errors::InvalidArgument(
                                "Could not parse entry in 'visible_device_list': '",
                                platform_gpu_id_str,
                                "'. visible_device_list = ", visible_device_list);
                    }
                    if (platform_gpu_id < 0 ||
                            platform_gpu_id >= gpu_manager->VisibleDeviceCount()) {
                        return errors::InvalidArgument(
                                "'visible_device_list' listed an invalid GPU id '", platform_gpu_id,
                                "' but visible device count is ",
                                gpu_manager->VisibleDeviceCount());
                    }
                    visible_gpu_order->push_back(PlatformGpuId(platform_gpu_id));
                }
            }
            // Validate no repeats.
            std::set<PlatformGpuId> visible_device_set(visible_gpu_order->begin(),
                    visible_gpu_order->end());
            if (visible_device_set.size() != visible_gpu_order->size()) {
                return errors::InvalidArgument(
                        "visible_device_list contained a duplicate entry: ",
                        visible_device_list);
            }
            return Status::OK();
        }

        Status VerifyVirtualDeviceSettings(
                const size_t num_gpus_to_use, const GPUOptions& gpu_options,
                const std::vector<PlatformGpuId>& visible_gpu_order,
                const std::vector<PlatformGpuId>& valid_platform_gpu_ids) {
            const auto& virtual_devices = gpu_options.experimental().virtual_devices();
            CHECK(!virtual_devices.empty());
            if (gpu_options.per_process_gpu_memory_fraction() > 0) {
                return errors::InvalidArgument(
                        "It's invalid to set per_process_gpu_memory_fraction when "
                        "virtual_devices is set.");
            }
            if (num_gpus_to_use < virtual_devices.size()) {
                return errors::Unknown(
                        "Not enough GPUs to create virtual devices."
                        " num_gpus_to_use: ",
                        num_gpus_to_use, " #virtual_devices: ", virtual_devices.size());
            }

            if (!gpu_options.visible_device_list().empty() &&
                    visible_gpu_order.size() != virtual_devices.size()) {
                return errors::InvalidArgument(
                        "The number of GPUs in visible_device_list doesn't match the number "
                        "of elements in the virtual_devices list.",
                        " #GPUs in visible_device_list: ", visible_gpu_order.size(),
                        " virtual_devices.size(): ", virtual_devices.size());
            }
            if (valid_platform_gpu_ids.size() != virtual_devices.size()) {
                return errors::Unknown(
                        "The number of valid GPUs doesn't match the number of elements in "
                        "the virtual_devices list.",
                        " #valid GPUs: ", valid_platform_gpu_ids.size(),
                        " virtual_devices.size(): ", virtual_devices.size());
            }
            return Status::OK();
        }

        int64 MinSystemMemory(int64 available_memory) {
            // We use the following heuristic for now:
            //
            // If the available_memory is < 2GiB, we allocate 225MiB to system memory.
            // Otherwise, allocate max(300MiB, 0.05 * available_memory) to system memory.
            //
            // In the future we could be more sophisticated by using a table of devices.
            int64 min_system_memory;
            if (available_memory < (1LL << 31)) {
                // 225MiB
                min_system_memory = 225 * 1024 * 1024;
            } else {
                // max(300 MiB, 0.05 * available_memory)
                min_system_memory =
                    std::max(int64{314572800}, static_cast<int64>(available_memory * 0.05));
            }

            VLOG(5) << "available_memory = " << available_memory;
            VLOG(5) << "min_system_memory = " << min_system_memory;
            return min_system_memory;
        }

        // Get the memory limit for the virtual device being created on GPU with
        // 'platform_gpu_id', when that virtual device is the only virtual device being
        // created on that GPU.
        Status SingleVirtualDeviceMemoryLimit(const GPUOptions& gpu_options,
                PlatformGpuId platform_gpu_id,
                int64* memory_limit) {

            int64 total_memory = 0;
            int64 available_memory = 0;
            se::StreamExecutor* se =
                GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie();
            // if (!se->DeviceMemoryUsage(&available_memory, &total_memory)) {
            // return errors::Unknown("Failed to query available memory for GPU ",
            // platform_gpu_id);
            // }
            available_memory = 225 * 1024 * 1024;

            int64 allocated_memory = 0;
            const double per_process_gpu_memory_fraction =
                gpu_options.per_process_gpu_memory_fraction();
            if (per_process_gpu_memory_fraction > 1.0 ||
                    gpu_options.experimental().use_unified_memory()) {
                // check device support unified memory
            }

            if (per_process_gpu_memory_fraction == 0) {
                // use all left gpu memory, sys mem is not included by memory_limit
                allocated_memory = available_memory;
                const int64 min_system_memory = MinSystemMemory(available_memory);
                if (min_system_memory < allocated_memory) {
                    allocated_memory -= min_system_memory;
                }
            } else {
                allocated_memory = total_memory * per_process_gpu_memory_fraction;
            }
            *memory_limit = allocated_memory;
            return Status::OK();
        }
    } // namespace

    Status BaseGPUDeviceFactory::GetDeviceLocalities(
            int num_tf_gpus, LocalityMap* localities) {
        std::vector<TfGpuId> all_tf_gpu_ids;
        all_tf_gpu_ids.reserve(num_tf_gpus);
        for (int i = 0; i < num_tf_gpus; ++i) {
            all_tf_gpu_ids.push_back(TfGpuId(i));
        }

        for (TfGpuId tf_gpu_id : all_tf_gpu_ids) {
            PlatformGpuId platform_gpu_id;
            TF_RETURN_IF_ERROR(
                    GpuIdManager::TfToPlatformGpuId(tf_gpu_id, &platform_gpu_id));
            // Get GPU bus_id from its reported NUMA affinity.  Because GPUs are
            // virtualized in some environments, we can't just use the GPU id.
            // NUMA locales are indexed from 0, buses are indexed from 1.
            se::Platform* gpu_manager = GPUMachineManager();
            auto desc_status =
                gpu_manager->DescriptionForDevice(platform_gpu_id);
            if (!desc_status.ok()) {
                return desc_status.status();
            }
            auto desc = desc_status.ConsumeValueOrDie();
            int numa_node = desc->numa_node();

            if (numa_node < 0) {
                // For some reason the StreamExecutor couldn't get the NUMA
                // affinity of the GPU.  If this is not a multi-socket mobo with
                // GPUs local to different buses, it doesn't matter.  If it is, we
                // may run into trouble later with data transfer operations.  The
                // trouble may manifest as slower than expected performance, or
                // outright failures.
                LOG(INFO) << "Could not identify NUMA node of platform GPU id "
                    << platform_gpu_id
                    << ", defaulting to 0.  Your kernel may not have been built "
                    << "with NUMA support.";
                numa_node = 0;
            }
            DeviceLocality dev_locality;
            dev_locality.set_numa_node(numa_node);
            dev_locality.set_bus_id(numa_node + 1);

            (*localities)[tf_gpu_id] = dev_locality;
            VLOG(1) << "GPUDevice PlatformGpuId " << platform_gpu_id << " TfGpuId "
                << tf_gpu_id << " on bus " << dev_locality.bus_id()
                << " numa: " << numa_node << " pci: " << desc->pci_bus_id()
                << " DeviceLocality: " << dev_locality.DebugString();
        }

        return Status::OK();
    }


    // device factory
    Status BaseGPUDeviceFactory::CreateDevices(
            const SessionOptions& options, const string& name_prefix,
            std::vector<std::unique_ptr<Device>>* devices) {
        TF_RETURN_IF_ERROR(ValidateGPUMachineManager());
        se::Platform* gpu_manager = GPUMachineManager();
        if (gpu_manager == nullptr) {
            return Status::OK();
        }
        // If there are no GPUs visible, do nothing.
        if (gpu_manager->VisibleDeviceCount() <= 0) {
            return Status::OK();
        }

        size_t num_gpus_to_use = INT_MAX;
        auto iter = options.config.device_count().find("GPU");
        if (iter != options.config.device_count().end()) {
            num_gpus_to_use = iter->second;
        }

        const auto& gpu_options = options.config.gpu_options();
        std::vector<PlatformGpuId> visible_gpu_order;
        std::vector<PlatformGpuId> valid_platform_gpu_ids;
        // If we aren't going to use any GPUs, don't initialize them.
        // We don't want to call ParseVisibleDeviceList if num_gpus_to_use is 0,
        // because it treats an empty gpu_options.visible_device_list as 'all GPUs
        // are visible'.
        if (num_gpus_to_use > 0) {
            TF_RETURN_IF_ERROR(ParseVisibleDeviceList(gpu_options.visible_device_list(),
                        &visible_gpu_order));
            bool new_gpu_found = false;
            for (int i = 0; i < visible_gpu_order.size(); ++i) {
                int visible_gpu_id = visible_gpu_order[i];

                // Only perform this once per visible gpu id.
                if (visible_gpu_initialized_[visible_gpu_id]) {
                    continue;
                }

                visible_gpu_initialized_[visible_gpu_id] = true;
                new_gpu_found = true;
            }

            TF_RETURN_IF_ERROR(
                    GetValidDeviceIds(visible_gpu_order, &valid_platform_gpu_ids));
        }

        if (num_gpus_to_use > valid_platform_gpu_ids.size()) {
            num_gpus_to_use = valid_platform_gpu_ids.size();
        }

        if (!valid_platform_gpu_ids.empty()) {
            // Save the original device.
            int original_device = 0;
        }

        const auto& virtual_devices = gpu_options.experimental().virtual_devices();
        if (!virtual_devices.empty()) {
            TF_RETURN_IF_ERROR(VerifyVirtualDeviceSettings(num_gpus_to_use, gpu_options,
                        visible_gpu_order,
                        valid_platform_gpu_ids));
            // We've verified that num_gpus_to_use >= virtual_devices.size().
            num_gpus_to_use = virtual_devices.size();
            CHECK(gpu_options.visible_device_list().empty() ||
                    valid_platform_gpu_ids == visible_gpu_order);
        }

        // including virtual devices
        int next_gpu_id = 0;
        std::vector<int64> memory_limit_bytes;
        for (int i = 0; i < num_gpus_to_use; ++i) {
            const PlatformGpuId platform_gpu_id = valid_platform_gpu_ids[i];
            if (virtual_devices.empty() ||
                    virtual_devices.Get(i).memory_limit_mb_size() == 0) {
                int64 single_virtual_device_memory_limit = 0;
                TF_RETURN_IF_ERROR(SingleVirtualDeviceMemoryLimit(
                            gpu_options, platform_gpu_id, &single_virtual_device_memory_limit));
                memory_limit_bytes.push_back(single_virtual_device_memory_limit);
            } else {
                const auto& memory_limit_mb = virtual_devices.Get(i).memory_limit_mb();
                std::transform(memory_limit_mb.begin(), memory_limit_mb.end(),
                        std::back_inserter(memory_limit_bytes), [](float mb) {
                        return static_cast<int64>(mb) * (1ll << 20);
                        });
            }

            while (next_gpu_id < memory_limit_bytes.size()) {
                TfGpuId gpu_id(next_gpu_id);
                ++next_gpu_id;
                TF_RETURN_IF_ERROR(
                        GpuIdManager::InsertTfPlatformGpuIdPair(gpu_id, platform_gpu_id));
            }
        }

        const int num_gpus = next_gpu_id;
        LocalityMap device_localities;
        TF_RETURN_IF_ERROR(
                GetDeviceLocalities(num_gpus, &device_localities));

        for (int di = 0; di < num_gpus; ++di) {
            TfGpuId tf_gpu_id(di);
            int64 bytes = memory_limit_bytes[di];
            auto it = device_localities.find(tf_gpu_id);
            if (it == device_localities.end()) {
                return errors::Internal("Failed to find DeviceLocality for GPU device ",
                        tf_gpu_id);
            }
            TF_RETURN_IF_ERROR(CreateGPUDevice(options, name_prefix, tf_gpu_id, bytes,
                        it->second, devices));
        }

        return Status::OK();
    }

    // static string GetShortDeviceDescription(PlatformGpuId platform_gpu_id,
    // const se::DeviceDescription& desc) {
    // int cc_major;
    // int cc_minor;
    // if (!desc.cuda_compute_capability(&cc_major, &cc_minor)) {
    // cc_major = 0;
    // cc_minor = 0;
    // }
    // // LINT.IfChange
    // return strings::StrCat("device: ", platform_gpu_id.value(),
    // ", name: ", desc.name(),
    // ", pci bus id: ", desc.pci_bus_id(),
    // ", compute capability: ", cc_major, ".", cc_minor);
    // // LINT.ThenChange(//tensorflow/python/platform/test.py)
    // }

    Status BaseGPUDeviceFactory::CreateGPUDevice(
            const SessionOptions& options, const string& name_prefix, TfGpuId tf_gpu_id,
            int64 memory_limit, const DeviceLocality& dev_locality,
            std::vector<std::unique_ptr<Device>>* devices) {
        CHECK_GE(tf_gpu_id, 0);
        const string device_name =
            strings::StrCat(name_prefix, "/device:GPU:", tf_gpu_id);

        PlatformGpuId platform_gpu_id;
        TF_RETURN_IF_ERROR(
                GpuIdManager::TfToPlatformGpuId(tf_gpu_id, &platform_gpu_id));
        int numa_node = dev_locality.numa_node();

        // get gpu allocator from gpu ps
        GPUProcessState* process_state = GPUProcessState::singleton();
        Allocator* gpu_allocator = process_state->GetGPUAllocator(
                options.config.gpu_options(), tf_gpu_id, memory_limit);
        if (gpu_allocator == nullptr) {
            return errors::Internal("Failed to get memory allocator for TF GPU ",
                    tf_gpu_id, " with ", memory_limit,
                    " bytes of memory.");
        }
        absl::optional<AllocatorStats> stats = gpu_allocator->GetStats();
        if (!stats) {
            return errors::Internal("No allocator statistics");
        }

        // 'memory_limit' is the required memory size, but if the allocator with
        // given tf_gpu_id was created before, we'll use it instead of creating a
        // new one (as TF gpu device is a shared resource), in which case the actual
        // memory limit represented by 'stats.bytes_limit' used by that allocator
        // may be different (which should be an error).
        //
        // TODO(laigd): report error if memory_limit doesn't match
        // stats->bytes_limit.
        int64 bytes_limit = stats->bytes_limit ? *stats->bytes_limit : 0;
        std::unique_ptr<BaseGPUDevice> gpu_device = CreateGPUDevice(
                options, device_name, static_cast<Bytes>(bytes_limit), dev_locality,
                tf_gpu_id, "",
                gpu_allocator, ProcessState::singleton()->GetCPUAllocator(numa_node));
        LOG(INFO) << "Created TensorFlow device (" << device_name << " with "
            << (bytes_limit >> 20) << " MB memory) -> physical GPU ("
            << "" << ")";
        TF_RETURN_IF_ERROR(gpu_device->Init(options));
        devices->push_back(std::move(gpu_device));

        return Status::OK();
    }

    Status BaseGPUDeviceFactory::GetValidDeviceIds(
            const std::vector<PlatformGpuId>& visible_gpu_order,
            std::vector<PlatformGpuId>* ids) {
        se::Platform* gpu_manager = GPUMachineManager();
        for (int i = 0; i < visible_gpu_order.size(); ++i) {
            int visible_gpu_id = visible_gpu_order[i];
            auto description_status = gpu_manager->DescriptionForDevice(visible_gpu_id);
            if (!description_status.ok()) {
                return description_status.status();
            }

            auto description = description_status.ConsumeValueOrDie();
            // check gpu version and its computeCapability
            LOG(INFO) << "Found device " << i << " with properties: "
                << "\npciBusID: " << description->pci_bus_id()
                << " name: " << description->name();
        }

        // Filter out devices that don't have the right capability or power.
        for (int i = 0; i < visible_gpu_order.size(); ++i) {
            const PlatformGpuId visible_gpu_id = visible_gpu_order[i];
            auto description_status =
                gpu_manager->DescriptionForDevice(visible_gpu_id);
            if (!description_status.ok()) {
                LOG(INFO) << "Ignoring visible gpu device " << visible_gpu_id
                    << " whose executor is in invalid state: "
                    << description_status.status().ToString();
                continue;
            }

            auto desc = description_status.ConsumeValueOrDie();

            // here we just check nothing to use all ids
            ids->push_back(visible_gpu_id);
        }

        if (!ids->empty()) {
            std::vector<int> raw_ids(ids->size());
            std::transform(ids->begin(), ids->end(), raw_ids.begin(),
                    [](PlatformGpuId id) -> int { return id; });
            LOG(INFO) << "Adding visible gpu devices: " << absl::StrJoin(raw_ids, ", ");
        }
        return Status::OK();
    }
}// namespace dlxnet
