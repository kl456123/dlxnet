#ifndef DLXNET_STREAM_EXECUTOR_LIB_STATUS_H_
#define DLXNET_STREAM_EXECUTOR_LIB_STATUS_H_
#include "absl/strings/string_view.h"
#include "dlxnet/core/lib/core/status.h"
#include "dlxnet/stream_executor/lib/error.h"  // IWYU pragma: export
#include "dlxnet/stream_executor/platform/logging.h"


namespace stream_executor {
    namespace port {

        using Status = dlxnet::Status;

#define SE_CHECK_OK(val) TF_CHECK_OK(val)
#define SE_ASSERT_OK(val) \
        ASSERT_EQ(::stream_executor::port::Status::OK(), (val))

        // Define some canonical error helpers.
        inline Status UnimplementedError(absl::string_view message) {
            return Status(error::UNIMPLEMENTED, message);
        }
        inline Status InternalError(absl::string_view message) {
            return Status(error::INTERNAL, message);
        }
        inline Status FailedPreconditionError(absl::string_view message) {
            return Status(error::FAILED_PRECONDITION, message);
        }

    }  // namespace port
}

#endif
