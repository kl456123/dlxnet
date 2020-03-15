#ifndef DLXNET_CORE_FRAMEWORK_SESSION_STATE_H_
#define DLXNET_CORE_FRAMEWORK_SESSION_STATE_H_
#include <string>
#include <unordered_map>
#include <vector>

#include "dlxnet/core/framework/tensor.h"
#include "dlxnet/core/lib/errors.h"
#include "dlxnet/core/platform/mutex.h"

namespace dlxnet{
    // The session state remembers the tensors we choose to keep across
    // multiple run calls.
    class SessionState {
        public:
            // Get a tensor from the session state.
            Status GetTensor(const string& handle, Tensor* tensor);

            // Store a tensor in the session state.
            Status AddTensor(const string& handle, const Tensor& tensor);

            // Delete a tensdor from the session state.
            Status DeleteTensor(const string& handle);

            int64 GetNewId();

            static const char* kTensorHandleResourceTypeName;

        private:
            mutex state_lock_;

            // For generating unique ids for tensors stored in the session.
            int64 tensor_id_ = 0;

            // The live tensors in the session. A map from tensor handle to tensor.
            std::unordered_map<string, Tensor> tensors_;
    };
}


#endif
