#include "dlxnet/core/common_runtime/executor.h"
#include "dlxnet/core/common_runtime/executor_factory.h"
#include "dlxnet/core/framework/op_kernel.h"
#include "dlxnet/core/platform/mutex.h"
#include "dlxnet/core/lib/gtl/inlined_vector.h"
#include "dlxnet/core/lib/gtl/manual_constructor.h"
#include "dlxnet/core/framework/log_memory.h"
#include "dlxnet/core/lib/gtl/flatmap.h"


namespace dlxnet{
    class ExecutorImpl;
    class GraphView;
    class NodeItem;

    typedef std::vector<int> PendingCounts;

    struct EdgeInfo {
        int dst_id;
        int output_slot : 31;
        // true if this is the last info for output_slot in the EdgeInfo list.
        bool is_last : 1;
        int input_slot;
    };

    // Compact structure representing a graph node and its associated kernel.
    //
    // Each NodeItem is an element of exactly one GraphView.
    struct NodeItem {
        NodeItem() {}
        // The index of this node's item in its GraphView.
        int node_id = -1;
        // Cached attributes of this node for fast lookup.
        bool kernel_is_async : 1;     // True iff kernel->AsAsync() != nullptr
        bool is_source : 1;           // True iff IsSource(node)
        bool is_sink : 1;             // True iff IsSink(node)

        // The kernel for this node.
        OpKernel* kernel = nullptr;

        // Cached values of node->num_inputs() and node->num_outputs(), to
        // avoid levels of indirection.
        int num_inputs;
        int num_outputs;

        // ExecutorImpl::tensors_[input_start] is the 1st positional input
        // for this node.
        int input_start = 0;

        // Number of output edges.
        size_t num_output_edges;

        const EdgeInfo* output_edge_list() const { return output_edge_base(); }

        // ith output edge.
        const EdgeInfo& output_edge(int i) const {
            DCHECK_GE(i, 0);
            DCHECK_LT(i, num_output_edges);
            return output_edge_base()[i];
        }

        DataType input_type(int i) const {
            DCHECK_LT(i, num_inputs);
            return static_cast<DataType>(input_type_base()[i]);
        }
        DataType output_type(int i) const {
            DCHECK_LT(i, num_outputs);
            return static_cast<DataType>(output_type_base()[i]);
        }

        // Return array of per-output allocator attributes.
        const AllocatorAttributes* output_attr_list() const { return output_attr_base(); }

        string DebugString() const {
            string ret = strings::StrCat("{name:'", kernel->name(), "' id:", node_id);
            if (is_source) {
                strings::StrAppend(&ret, " source}");
            } else if (is_sink) {
                strings::StrAppend(&ret, " sink}");
            } else {
                strings::StrAppend(&ret, " def:{", SummarizeNodeDef(kernel->def()), "}}");
            }
            return ret;
        }


        private:
        friend class GraphView;

        uint8* input_type_base()const{
            return const_cast<uint8*>(input_types.data());
        }
        uint8* output_type_base()const{
            return const_cast<uint8*>(output_types.data());
        }

        AllocatorAttributes* output_attr_base()const{
            return const_cast<AllocatorAttributes*>(output_attrs.data());
        }
        EdgeInfo* output_edge_base()const{
            return const_cast<EdgeInfo*>(output_edges.data());
        }
        std::vector<uint8> input_types;
        std::vector<uint8> output_types;
        std::vector<AllocatorAttributes> output_attrs;
        std::vector<EdgeInfo> output_edges;
        TF_DISALLOW_COPY_AND_ASSIGN(NodeItem);
    };

    class GraphView{
        public:
            GraphView(){};
            ~GraphView();
            void Initialize(const Graph* g);
            NodeItem* node(size_t id)const{
                return node_items_[id];
            }
            int32 num_nodes() const { return node_items_.size(); }
        private:
            std::vector<NodeItem*> node_items_;
            void InitializeNode(NodeItem* node_item, const Node* n);
            TF_DISALLOW_COPY_AND_ASSIGN(GraphView);
    };

    // ExecutorImpl
    class ExecutorImpl: public Executor{
        public:
            explicit ExecutorImpl(const LocalExecutorParams& p) : params_(p), gview_(){
                CHECK(p.create_kernel != nullptr);
                CHECK(p.delete_kernel != nullptr);
            }
            ~ExecutorImpl() override {
                for (auto fiter : frame_info_) {
                    delete fiter.second;
                }
            }
            Status Initialize(const Graph& graph);
            void RunAsync(const Args& args, DoneCallback done) override;
        private:
            friend class ExecutorState;

            struct FrameInfo{
                FrameInfo()
                    :total_inputs(0),
                    input_count(0),
                    pending_counts(nullptr){}

                int input_count;

                int total_inputs;
                // Each frame has its own PendingCounts only for the nodes in the frame.
                PendingCounts* pending_counts;  // Owned

                ~FrameInfo(){}
            };

            void InitializePending(const Graph* graph);
            FrameInfo* EnsureFrameInfo(const string& fname){
                auto slot = &frame_info_[fname];
                if (*slot == nullptr) {
                    *slot = new FrameInfo;
                }
                return *slot;
            }

            // Owned.
            LocalExecutorParams params_;
            GraphView gview_;

            // Root nodes (with no in edges) that should form the initial ready queue
            std::vector<const NodeItem*> root_nodes_;

            // Mapping from frame name to static information about the frame.
            // TODO(yuanbyu): We could cache it along with the graph so to avoid
            // the overhead of constructing it for each executor instance.
            gtl::FlatMap<string, FrameInfo*> frame_info_;

            TF_DISALLOW_COPY_AND_ASSIGN(ExecutorImpl);
    };

    typedef gtl::InlinedVector<TensorValue, 4> TensorValueVec;
    typedef gtl::InlinedVector<AllocatorAttributes, 4> AllocatorAttributeVec;

    // The state associated with one invocation of ExecutorImpl::Run.
    // ExecutorState dispatches nodes when they become ready and keeps
    // track of how many predecessors of a node have not done (pending_).
    class ExecutorState{
        public:
            ExecutorState(const Executor::Args& args, ExecutorImpl* impl);
            ~ExecutorState();
            void RunAsync(Executor::DoneCallback done);
        private:
            // Either a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
            // TODO(yuanbyu): A better way to do "has_value"?
            struct Entry {
                Entry() {}
                Entry(const Entry& other)
                    : ref(other.ref),
                    ref_mu(other.ref_mu),
                    has_value(other.has_value),
                    val_field_is_set(other.val_field_is_set),
                    alloc_attr(other.alloc_attr) {
                        if (val_field_is_set) {
                            val.Init(*other.val);
                        }
                    }
                ~Entry() {
                    if (val_field_is_set) val.Destroy();
                }

                Entry& operator=(const Entry& other) {
                    if (val_field_is_set) {
                        val.Destroy();
                    }
                    ref = other.ref;
                    ref_mu = other.ref_mu;
                    has_value = other.has_value;
                    val_field_is_set = other.val_field_is_set;
                    alloc_attr = other.alloc_attr;
                    if (val_field_is_set) {
                        val.Init(*other.val);
                    }
                    return *this;
                }

                Entry& operator=(Entry&& other) {
                    if (val_field_is_set) {
                        val.Destroy();
                    }
                    ref = other.ref;
                    ref_mu = other.ref_mu;
                    has_value = other.has_value;
                    val_field_is_set = other.val_field_is_set;
                    alloc_attr = other.alloc_attr;
                    if (val_field_is_set) {
                        val.Init(std::move(*other.val));
                    }
                    return *this;
                }

                // Clears the <val> field.
                void ClearVal() {
                    if (val_field_is_set) {
                        val.Destroy();
                        val_field_is_set = false;
                        has_value = false;
                    }
                }

                // A tensor value, if val_field_is_set.
                ManualConstructor<Tensor> val;

                Tensor* ref = nullptr;    // A tensor reference.
                mutex* ref_mu = nullptr;  // mutex for *ref if ref is not nullptr.

                // Whether the value exists, either in <val> or <ref>.
                bool has_value = false;

                bool val_field_is_set = false;

                // The attributes of the allocator that creates the tensor.
                AllocatorAttributes alloc_attr;
            };
            // Contains the device context assigned by the device at the beginning of a
            // step.
            DeviceContext* device_context_ = nullptr;

            struct TaggedNode;
            typedef gtl::InlinedVector<TaggedNode, 8> TaggedNodeSeq;
            typedef gtl::InlinedVector<Entry, 4> EntryVector;

            struct IterationState{
                explicit IterationState(PendingCounts* pending_counts,
                        int total_input_tensors)
                    : input_tensors(new Entry[total_input_tensors]),
                    counts(pending_counts){}
                ~IterationState() { delete[] input_tensors; }
                Entry*  input_tensors;
                void adjust_for_activation(int h, int* pending_result,
                        int* dead_result) {
                    int* current_pending_count = &(*counts)[h];
                    CHECK_GT(*current_pending_count, 0);
                    (*current_pending_count)--;
                    *pending_result = *current_pending_count;
                }
                private:
                PendingCounts* counts;
            };

            struct FrameState{
                explicit FrameState(const ExecutorImpl* impl)
                    :executor(impl){}
                const ExecutorImpl* executor = nullptr;

                string frame_name;
                uint64 frame_id;

                // The number of inputs this frame is still waiting.
                int num_pending_inputs = 0;
                int total_input_tensors = 0;
                PendingCounts* pending_counts=nullptr;
                private:
                // The active iteration states of this frame.
                IterationState* iteration_first GUARDED_BY(mu);

                public:
                // Lock ordering: ExecutorState.mu_ < mu;
                // during structured traversal: parent_frame->mu < mu.
                mutex mu;

                void InitializeFrameInfo(const string& enter_name) {
                    auto it_frame_info = executor->frame_info_.find(enter_name);
                    DCHECK(it_frame_info != executor->frame_info_.end());
                    ExecutorImpl::FrameInfo* finfo = it_frame_info->second;
                    total_input_tensors = finfo->total_inputs;
                    num_pending_inputs = finfo->input_count;
                    pending_counts = finfo->pending_counts;
                }

                inline IterationState* GetIteration(int64 iter)
                    EXCLUSIVE_LOCKS_REQUIRED(mu) {
                        return iteration_first;
                    }

                inline void SetIteration(IterationState* state)
                    EXCLUSIVE_LOCKS_REQUIRED(mu){
                        iteration_first = state;
                    }

                // Activate the successors of a node. Contents of *outputs are left in an
                // indeterminate state after returning from this method.
                void ActivateNodes(const NodeItem* item, const bool is_dead, int64 iter,
                        EntryVector* outputs, TaggedNodeSeq* ready)
                    EXCLUSIVE_LOCKS_REQUIRED(mu);


                ~FrameState() {
                    delete iteration_first;
                    iteration_first = nullptr;
                }

            };



            // A tagged node: <frame*, iter, node*>.
            struct TaggedNode {
                const NodeItem* node_item;
                bool is_dead = false;
                int64 input_iter = -1;
                FrameState* input_frame;

                TaggedNode(const NodeItem* node_item, FrameState* in_frame, int64 in_iter,
                        bool dead)
                    : node_item(node_item),
                    input_frame(in_frame),
                    input_iter(in_iter),
                    is_dead(dead) {}
            };

            // A drop-in replacement for std::deque<TaggedNode>.  We typically don't
            // have that many nodes in the ready queue, so we just use a vector and
            // don't free up memory from the queue as we consume nodes.
            class TaggedNodeReadyQueue {
                public:
                    TaggedNodeReadyQueue() : front_index_(0) {}

                    void push_back(TaggedNode node) { ready_.push_back(node); }
                    TaggedNode front() const {
                        DCHECK_LT(front_index_, ready_.size());
                        return ready_[front_index_];
                    }
                    void pop_front() {
                        DCHECK_LT(front_index_, ready_.size());
                        front_index_++;
                        if ((front_index_ == ready_.size()) || (front_index_ > 16384)) {
                            if (front_index_ == ready_.size()) {
                                ready_.clear();
                            } else {
                                // Lots of unused entries at beginning of vector: move everything
                                // down to start of vector.
                                ready_.erase(ready_.begin(), ready_.begin() + front_index_);
                            }
                            front_index_ = 0;
                        }
                    }
                    bool empty() const { return ready_.empty(); }
                    const TaggedNode* begin() const { return ready_.begin() + front_index_; }
                    const TaggedNode* end() const { return ready_.end(); }

                private:
                    gtl::InlinedVector<TaggedNode, 16> ready_;
                    int front_index_;
            };
            const bool vlog_;  // true if VLOG_IS_ON(1). Used to check vlog cheaply.

            // true if LogMemory::IsEnabled(). Used to check memory enabled cheaply.
            const bool log_memory_;
            SessionState* session_state_;
            string session_handle_;
            CallFrameInterface* call_frame_;
            const ExecutorImpl* impl_;
            Executor::Args::Runner runner_;
            bool sync_on_finish_;
            int64 step_id_;

            // Owned.
            mutex mu_;
            Status status_ GUARDED_BY(mu_);

            // The root frame in which the execution of this step is started.
            FrameState* root_frame_;

            // Invoked when the execution finishes.
            Executor::DoneCallback done_cb_;
            std::atomic_int_fast32_t num_outstanding_ops_;

            // Process a ready node in current thread.
            void Process(TaggedNode node);

            // Before invoking item->kernel, fills in its "inputs".
            Status PrepareInputs(const NodeItem& item, Entry* first_input,
                    TensorValueVec* inputs,
                    AllocatorAttributeVec* input_alloc_attrs,
                    bool* is_input_dead);

            // After item->kernel computation is done, processes its outputs.
            Status ProcessOutputs(const NodeItem& item, OpKernelContext* ctx,
                    EntryVector* outputs);

            // After processing the outputs, propagates the outputs to their dsts.
            // Contents of *outputs are left in an indeterminate state after
            // returning from this method.
            void PropagateOutputs(const TaggedNode& tagged_node, const NodeItem* item,
                    EntryVector* outputs, TaggedNodeSeq* ready);

            // Called after each node finishes. Takes ownership of "stats". Returns true
            // if execution has completed.
            bool NodeDone(const Status& s, const TaggedNodeSeq& ready,
                    TaggedNodeReadyQueue* inline_ready);

            // Schedule all the expensive nodes in 'ready', and put all the inexpensive
            // nodes in 'ready' into 'inline_ready'.
            void ScheduleReady(const TaggedNodeSeq& ready,
                    TaggedNodeReadyQueue* inline_ready);

            // Clean up when this executor is done.
            void Finish();
            void ScheduleFinish();

            // A standalone routine for this expression so that we can express
            // that we don't want thread safety analysis on this reference (it's
            // safe to do without the lock because the iterations array never
            // resizes and this particular iteration's array element will not
            // be changed out from under us because the iteration is still alive).
            Entry* GetInputTensors(FrameState* input_frame,
                    int64 input_iter) const NO_THREAD_SAFETY_ANALYSIS {
                return input_frame->GetIteration(input_iter)->input_tensors;
            }
    };

    ExecutorState::~ExecutorState(){
        if (device_context_) {
            device_context_->Unref();
        }
    }
    void ExecutorState::Finish(){
        mu_.lock();
        auto status = status_;
        auto done_cb = std::move(done_cb_);
        auto runner = std::move(runner_);
        mu_.unlock();
        int64 step_id = step_id_;
        CHECK(done_cb != nullptr);
        Device* device = impl_->params_.device;

        delete this;
        runner([step_id, status, done_cb = std::move(done_cb)]() {
                done_cb(status);
                });
    }

    void ExecutorState::ScheduleFinish(){
        Finish();
    }

    void ExecutorState::Process(TaggedNode tagged_node){
        // current process just execute nodes in inline_ready
        // start from single node(tagged_node)
        TaggedNodeSeq ready;
        TaggedNodeReadyQueue inline_ready;

        // prepare params and then compute it
        OpKernelContext::Params params;

        Device* device = impl_->params_.device;
        params.device = device;

        // Parameters passed to OpKernel::Compute.
        TensorValueVec inputs;
        AllocatorAttributeVec input_alloc_attrs;

        params.log_memory = log_memory_;
        params.session_state = session_state_;
        params.session_handle = session_handle_;
        params.call_frame = call_frame_;
        params.inputs = &inputs;
        params.input_alloc_attrs = &input_alloc_attrs;
        params.runner = &runner_;

        // Set the device_context for this device, if it exists.
        params.op_device_context = device_context_;


        Status s;
        inline_ready.push_back(tagged_node);
        EntryVector outputs;
        bool completed = false;
        while(!inline_ready.empty()){
            tagged_node = inline_ready.front();
            inline_ready.pop_front();
            const NodeItem& item = *tagged_node.node_item;
            FrameState* input_frame = tagged_node.input_frame;
            const int64 input_iter = tagged_node.input_iter;
            const int id = item.node_id;

            if (vlog_) {
                VLOG(1) << "Process node: " << id << " step " << params.step_id << " "
                    << SummarizeNodeDef(item.kernel->def())
                    << (tagged_node.is_dead ? " is dead" : "")
                    << " device: " << device->name();
            }
            outputs.clear();
            DeviceContext* device_context = nullptr;

            Entry* input_tensors = GetInputTensors(input_frame, input_iter);
            Entry* first_input = input_tensors + item.input_start;
            bool launched_asynchronously = false;

            // Prepares inputs.
            bool is_input_dead = false;
            s = PrepareInputs(item, first_input, &inputs, &input_alloc_attrs,
                    &is_input_dead);

            if (!s.ok()) {
                // Clear inputs.
                int num_inputs = item.num_inputs;
                for (int i = 0; i < num_inputs; ++i) {
                    // (first_input + i)->ClearVal();
                }
                // MaybeMarkCompleted(input_frame, input_iter, item);
                // Continue to process the nodes in 'inline_ready'.
                completed = NodeDone(s, ready, &inline_ready);
                continue;
            }

            // Set up compute params.
            OpKernel* op_kernel = item.kernel;
            params.op_kernel = op_kernel;
            params.is_input_dead = is_input_dead;
            params.output_attr_array = item.output_attr_list();
            if (item.kernel_is_async) {
                // Asynchronous computes.
                launched_asynchronously = true;
                // not supported now
            }else{
                // Synchronous computes.
                OpKernelContext ctx(&params, item.num_outputs);
                device->Compute(op_kernel, &ctx);
                s = ProcessOutputs(item, &ctx, &outputs);
                device_context = ctx.op_device_context();
            }

            if (!launched_asynchronously) {
                if (vlog_) {
                    VLOG(2) << "Synchronous kernel done: " << id << " step "
                        << params.step_id << " " << SummarizeNodeDef(item.kernel->def())
                        << (tagged_node.is_dead ? " is dead: " : "")
                        << " device: " << device->name();
                }

                // Propagates outputs.
                if (s.ok()) {
                    PropagateOutputs(tagged_node, &item, &outputs, &ready);
                }
                outputs.clear();
                // Postprocess.
                completed = NodeDone(s, ready, &inline_ready);
            }
        }// while !inline_ready.empty()
        // This thread of computation is done if completed = true.
        if (completed) ScheduleFinish();
    }

    void ExecutorState::RunAsync(Executor::DoneCallback done){
        TaggedNodeSeq ready;
        // Ask the device to fill in the device context map.
        Device* device = impl_->params_.device;
        const Status get_context_status =
            device->TryGetDeviceContext(&device_context_);
        if (!get_context_status.ok()) {
            delete this;
            done(get_context_status);
            return;
        }

        // Initialize the ready queue.
        for (const NodeItem* item : impl_->root_nodes_) {
            DCHECK_EQ(item->num_inputs, 0);
            ready.push_back(TaggedNode{item, root_frame_, 0, false});
        }
        if (ready.empty()) {
            delete this;
            done(Status::OK());
        } else {
            num_outstanding_ops_ = ready.size();
            done_cb_ = std::move(done);
            // Schedule to run all the ready ops in thread pool.
            ScheduleReady(ready, nullptr);
        }
    }

    Status ExecutorState::PrepareInputs(const NodeItem& item, Entry* first_input,
            TensorValueVec* inputs, AllocatorAttributeVec* input_alloc_attrs,
            bool* is_input_dead){
        inputs->clear();
        inputs->resize(item.num_inputs);
        input_alloc_attrs->clear();
        input_alloc_attrs->resize(item.num_inputs);

        *is_input_dead = false;

        for (int i = 0; i < item.num_inputs; ++i){
            const bool expect_ref = IsRefType(item.input_type(i));
            Entry* entry = first_input + i;
            (*input_alloc_attrs)[i] = entry->alloc_attr;

            // i-th input.
            TensorValue* inp = &(*inputs)[i];

            // Only merge and transfer nodes can have no-value inputs.
            if (!entry->has_value) {
                continue;
            }
            if (entry->ref == nullptr) {
                if (expect_ref) {
                    return errors::InvalidArgument(i, "-th input expects a ref type");
                }
                inp->tensor = entry->val.get();
            }else{
                {
                    tf_shared_lock ml(*entry->ref_mu);
                    if (!entry->ref->IsInitialized()) {
                        return errors::FailedPrecondition(
                                "Attempting to use uninitialized value ",
                                item.kernel->requested_input(i));
                    }
                }
                inp->mutex_if_ref = entry->ref_mu;
                inp->tensor = entry->ref;
            }
        }
        return Status::OK();
    }

    void ExecutorState::PropagateOutputs(const TaggedNode& tagged_node, const NodeItem* item,
            EntryVector* outputs, TaggedNodeSeq* ready){
        FrameState* input_frame = tagged_node.input_frame;
        const int64 input_iter = tagged_node.input_iter;
        const bool is_dead = tagged_node.is_dead;

        // Propagates outputs along out edges, and puts newly ready nodes
        // into the ready queue.
        ready->clear();
        bool is_frame_done = false;
        FrameState* output_frame = input_frame;
        int64 output_iter = input_iter;

        // Fast path for nodes types that don't need special handling
        DCHECK_EQ(input_frame, output_frame);
        // Normal path for most nodes
        mutex_lock l(input_frame->mu);
        output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
    }

    bool ExecutorState::NodeDone(const Status& s, const TaggedNodeSeq& ready,
            TaggedNodeReadyQueue* inline_ready){
        int ready_size = ready.size();
        bool completed = false;
        if (ready_size == 0 || !s.ok()) {
            completed = (num_outstanding_ops_.fetch_sub(1) == 1);
        } else if (ready_size > 1) {
            num_outstanding_ops_.fetch_add(ready_size - 1, std::memory_order_relaxed);
        }

        if(s.ok()){
            ScheduleReady(ready, inline_ready);
        }
        return completed;
    }

    Status ExecutorState::ProcessOutputs(const NodeItem& item, OpKernelContext* ctx,
            EntryVector* outputs){
        DCHECK_EQ(outputs->size(), 0);
        outputs->resize(item.num_outputs);
        // check status
        Status s = ctx->status();
        if(!s.ok()){
            return s;
        }

        for(int i=0;i<item.num_outputs; ++i){
            const TensorValue val = ctx->release_output(i);
            if(val.tensor==nullptr){
                s.Update(errors::Internal("Missing ", i, "-th output from ",
                            FormatNodeDefForError(item.kernel->def())));
            }else{
                Entry* out = &((*outputs)[i]);

                // Set the allocator attributes of the output entry.
                out->alloc_attr = ctx->output_alloc_attr(i);

                // Sanity check of output tensor types. We need to inspect this safely as
                // we are in the tensor buffer.
                DataType dtype = val.dtype_safe();
                if(dtype==item.output_type(i)){
                    // assign to out
                    if(val.is_ref()){
                        out->has_value = true;
                        out->ref = val.tensor;
                        out->ref_mu = val.mutex_if_ref;
                        if (log_memory_) {
                            Tensor to_log;
                            {
                                // Dereference the tensor under the lock.
                                tf_shared_lock l(*out->ref_mu);
                                to_log = *out->ref;
                            }
                            LogMemory::RecordTensorOutput(ctx->op_kernel().name(),
                                    ctx->step_id(), i, to_log);
                        }
                    }else{
                        // NOTE that std::move is used here, so val.tensor goes to
                        // uninitialized state (val.tensor->IsInitialized return false).
                        DCHECK(!out->val_field_is_set);
                        out->has_value = true;
                        out->val_field_is_set = true;
                        out->val.Init(std::move(*val.tensor));
                        if (log_memory_) {
                            LogMemory::RecordTensorOutput(ctx->op_kernel().name(),
                                    ctx->step_id(), i, *out->val);
                        }
                    }
                }else{
                    s.Update(
                            errors::Internal("Output ", i, " of type ", DataTypeString(dtype),
                                " does not match declared output type ",
                                DataTypeString(item.output_type(i)), " for node ",
                                FormatNodeDefForError(item.kernel->def())));
                }

                if (!val.is_ref()) {
                    // If OpKernelContext returns outputs via pass-by-value, we
                    // don't need this trouble.
                    delete val.tensor;
                }
            }
            return s;
        }
        return Status::OK();
    }

    void ExecutorState::ScheduleReady(const TaggedNodeSeq& ready,
            TaggedNodeReadyQueue* inline_ready){
        // all nodes that are ready put in ready list
        // then schedule them in thread pool

        // no need to schedule
        if(ready.empty())return;

        if(inline_ready==nullptr){
            // parallel all of them
            // Schedule to run all the ready ops in thread pool.
            for (auto& tagged_node : ready) {
                runner_([=]() { Process(tagged_node); });
            }
            return;
        }

        const TaggedNode* curr_expensive_node = nullptr;
        for (auto& tagged_node : ready) {
            const NodeItem& item = *tagged_node.node_item;
            if (tagged_node.is_dead || !item.kernel->IsExpensive()) {
                // Inline this inexpensive node.
                inline_ready->push_back(tagged_node);
            } else {
                if (curr_expensive_node) {
                    // Dispatch to another thread since there is plenty of work to
                    // do for this thread.
                    runner_(std::bind(&ExecutorState::Process, this, *curr_expensive_node));
                }
                curr_expensive_node = &tagged_node;
            }
        }
        if (curr_expensive_node) {
            if (inline_ready->empty()) {
                inline_ready->push_back(*curr_expensive_node);
            } else {
                // There are inline nodes to run already. We dispatch this expensive
                // node to other thread.
                runner_(std::bind(&ExecutorState::Process, this, *curr_expensive_node));
            }
        }


    }

    ExecutorState::ExecutorState(const Executor::Args& args, ExecutorImpl* impl)
        : vlog_(VLOG_IS_ON(1)),
        log_memory_(LogMemory::IsEnabled()),
        step_id_(args.step_id),
        session_state_(args.session_state),
        session_handle_(args.session_handle),
        call_frame_(args.call_frame),
        impl_(impl),
        runner_(args.runner),
        sync_on_finish_(args.sync_on_finish),
        num_outstanding_ops_(0) {
            // We start the entire execution in iteration 0 of the root frame
            // so let us create the root frame and the state for iteration 0.
            // We assume root_frame_->frame_name.empty().
            root_frame_ = new FrameState(impl_);
            root_frame_->frame_id = 0;  // must be 0
            root_frame_->InitializeFrameInfo(root_frame_->frame_name);

            // Initialize iteration 0.
            root_frame_->SetIteration(
                    new IterationState(root_frame_->pending_counts,
                        root_frame_->total_input_tensors));
        }

    void ExecutorState::FrameState::ActivateNodes(const NodeItem* item,
            const bool is_dead, int64 iter,
            EntryVector* outputs,
            TaggedNodeSeq* ready){
        const GraphView& gview = executor->gview_;
        IterationState* iter_state = GetIteration(iter);
        const size_t num_output_edges = item->num_output_edges;
        const EdgeInfo* edges = item->output_edge_list();
        Entry* input_tensors = iter_state->input_tensors;
        for (size_t out_index = 0; out_index < num_output_edges; out_index++) {
            const EdgeInfo& e = edges[out_index];
            const int dst_id = e.dst_id;
            const NodeItem* dst_item = gview.node(dst_id);
            // const PendingCounts::Handle dst_pending_id = dst_item->pending_id;
            int dst_pending_id = dst_item->node_id;
            const int src_slot = e.output_slot;
            bool dst_ready = false;
            bool dst_dead = false;
            bool dst_need_input = true;

            int pending, dead;
            iter_state->adjust_for_activation(dst_pending_id, &pending, &dead);
            dst_dead = (dead > 0);
            dst_ready = (pending==0);

            if (dst_need_input) {
                const int dst_slot = e.input_slot;
                const int dst_loc = dst_item->input_start + dst_slot;
                input_tensors[dst_loc] = (*outputs)[src_slot];
            }

            // Add dst to the ready queue if it's ready
            if (dst_ready) {
                ready->emplace_back(dst_item, this, iter, dst_dead);
            }
        }
    }



    Status ExecutorImpl::Initialize(const Graph& graph){
        // Preprocess every node in the graph to create an instance of op
        // kernel for each node.
        gview_.Initialize(&graph);
        for (const Node* n : graph.nodes()) {
            const int id = n->id();
            NodeItem* item = gview_.node(id);
            item->node_id = id;

            Status s = params_.create_kernel(params_.device, n->def(), &item->kernel);
            if (!s.ok()) {
                item->kernel = nullptr;
                LOG(ERROR) << "Executor failed to create kernel. " << s;
                return s;
            }

            CHECK(item->kernel);
            item->kernel_is_async = (item->kernel->AsAsync() != nullptr);
            item->is_source = IsSource(n);
            item->is_sink = IsSink(n);

            // See if this node is a root node, and if so, add item to root_nodes_.
            if (n->in_edges().empty()) {
                root_nodes_.push_back(item);
            }
            const string frame_name = "";
            FrameInfo* frame_info = EnsureFrameInfo(frame_name);
            item->input_start = frame_info->total_inputs;
            frame_info->total_inputs += n->num_inputs();
        }

        InitializePending(&graph);
        return Status::OK();
    }

    void ExecutorImpl::RunAsync(const Args& args, DoneCallback done){
        (new ExecutorState(args, this))->RunAsync(std::move(done));
    }
    void ExecutorImpl::InitializePending(const Graph* graph){
        // only root frame is used
        const string frame_name = "";
        const int num_nodes = graph->num_node_ids();

        FrameInfo* finfo = EnsureFrameInfo(frame_name);
        PendingCounts* counts = new PendingCounts();
        DCHECK_EQ(finfo->pending_counts, nullptr);
        finfo->pending_counts = counts;

        counts->reserve(num_nodes);
        for (const Node* node: graph->nodes()) {
            int pending_count = node->num_inputs();
            counts->push_back(pending_count);
        }
    }

    // GraphView
    GraphView::~GraphView() {
        for(NodeItem* item: node_items_){
            delete item;
        }
    }
    void GraphView::InitializeNode(NodeItem* item, const Node* n){
        // allocate memory first
        CHECK_NOTNULL(item);

        const size_t num_output_edges = n->out_edges().size();
        const int num_inputs = n->num_inputs();
        const int num_outputs = n->num_outputs();

        item->num_inputs = num_inputs;
        item->num_outputs = num_outputs;
        item->num_output_edges = num_output_edges;

        // init input and output types
        // Fill output edges.
        // Keep track of the last EdgeInfo in the EdgeInfo array that references
        // a given output slot.  For all but the last, we need to do a copy of the
        // Tensor when propagating results downstream in the graph, but for the
        // last one, we can just do a move of the Tensor object to propagate it.
        gtl::InlinedVector<EdgeInfo*, 4> last_indices(num_outputs, nullptr);
        item->output_edges.reserve(num_output_edges);
        EdgeInfo* dst_edge = item->output_edge_base();
        for (auto e : n->out_edges()) {
            dst_edge->dst_id = e->dst()->id();
            CHECK_LE(e->src_output(), 0x3FFFFFFF);  // Must fit in 31 bits
            dst_edge->output_slot = e->src_output();
            dst_edge->is_last = false;
            const int output_slot = dst_edge->output_slot;
            if (output_slot >= 0) {
                last_indices[output_slot] = dst_edge;
            }
            dst_edge->input_slot = e->dst_input();
            dst_edge++;
        }

        for (EdgeInfo* edge_info : last_indices) {
            if (edge_info != nullptr) {
                edge_info->is_last = true;
            }
        }

        item->output_attrs.reserve(num_outputs);
        item->input_types.reserve(num_inputs);
        item->output_types.reserve(num_outputs);

        AllocatorAttributes* output_attrs = item->output_attr_base();
        for (int i = 0; i < num_outputs; i++) {
            new (&item->output_attrs[i]) AllocatorAttributes();
        }

        DCHECK_LT(DataType_MAX, 255);  // Must fit in uint8
        uint8* input_types = item->input_type_base();
        for (int i = 0; i < num_inputs; i++) {
            item->input_types[i] = static_cast<uint8>(n->input_type(i));
            DCHECK_EQ(item->input_type(i), n->input_type(i));
        }
        for (int i = 0; i < num_outputs; ++i) {
            item->output_types[i] = static_cast<uint8>(n->output_type(i));
            DCHECK_EQ(item->output_type(i), n->output_type(i));
        }
    }

    void GraphView::Initialize(const Graph* g){
        for(const Node* n:g->nodes()){
            node_items_.push_back(new NodeItem);
            InitializeNode(node_items_.back(), n);
        }
    }


    Status NewLocalExecutor(const LocalExecutorParams& params, const Graph& graph,
            Executor** executor) {
        ExecutorImpl* impl = new ExecutorImpl(params);
        const Status s = impl->Initialize(graph);
        if (s.ok()) {
            *executor = impl;
        } else {
            delete impl;
        }
        return s;
    }

    Status CreateNonCachedKernel(Device* device, const NodeDef& ndef, OpKernel** kernel) {
        const auto device_type = DeviceType(device->attributes().device_type());
        auto allocator = device->GetAllocator(AllocatorAttributes());
        return CreateOpKernel(device_type, device, allocator, ndef, kernel);
    }

    void DeleteNonCachedKernel(OpKernel* kernel) { delete kernel; }
    namespace {

        class DefaultExecutorRegistrar {
            public:
                DefaultExecutorRegistrar() {
                    Factory* factory = new Factory;
                    ExecutorFactory::Register("", factory);
                    ExecutorFactory::Register("DEFAULT", factory);
                }

            private:
                class Factory : public ExecutorFactory {
                    Status NewExecutor(const LocalExecutorParams& params, const Graph& graph,
                            std::unique_ptr<Executor>* out_executor) override {
                        Executor* ret = nullptr;
                        TF_RETURN_IF_ERROR(NewLocalExecutor(params, std::move(graph), &ret));
                        out_executor->reset(ret);
                        return Status::OK();
                    }
                };
        };
        static DefaultExecutorRegistrar registrar;

    }  // namespace
} // namespace dlxnet
