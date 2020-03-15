#include "dlxnet/core/framework/session_state.h"
// #include "dlxnet/core/graph/tensor_id.h"


namespace dlxnet{
    const char* SessionState::kTensorHandleResourceTypeName = "TensorHandle";

    Status SessionState::GetTensor(const string& handle, Tensor* tensor) {
        mutex_lock l(state_lock_);
        auto it = tensors_.find(handle);
        if (it == tensors_.end()) {
            return errors::InvalidArgument("The tensor with handle '", handle,
                    "' is not in the session store.");
        }
        *tensor = it->second;
        return Status::OK();
    }

    Status SessionState::AddTensor(const string& handle, const Tensor& tensor) {
        mutex_lock l(state_lock_);
        if (!tensors_.insert({handle, tensor}).second) {
            return errors::InvalidArgument("Failed to add a tensor with handle '",
                    handle, "' to the session store.");
        }
        return Status::OK();
    }

    Status SessionState::DeleteTensor(const string& handle) {
        mutex_lock l(state_lock_);
        if (tensors_.erase(handle) == 0) {
            return errors::InvalidArgument("Failed to delete a tensor with handle '",
                    handle, "' in the session store.");
        }
        return Status::OK();
    }

    int64 SessionState::GetNewId() {
        mutex_lock l(state_lock_);
        return tensor_id_++;
    }

}
