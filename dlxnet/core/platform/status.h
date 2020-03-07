/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef DLXNET_CORE_PLATFORM_STATUS_H_
#define DLXNET_CORE_PLATFORM_STATUS_H_

#include <functional>
#include <iosfwd>
#include <memory>
#include <string>

#include "dlxnet/core/platform/logging.h"
#include "dlxnet/core/platform/macros.h"
#include "dlxnet/core/platform/stringpiece.h"
#include "dlxnet/core/platform/types.h"
#include "dlxnet/core/protobuf/error_codes.pb.h"

namespace dlxnet {

#if defined(__clang__)
// Only clang supports warn_unused_result as a type annotation.
class DLXNET_MUST_USE_RESULT Status;
#endif

/// @ingroup core
/// Denotes success or failure of a call in Tensorflow.
class Status {
 public:
  /// Create a success status.
  Status() {}

  /// \brief Create a status with the specified error code and msg as a
  /// human-readable string containing more detailed information.
  Status(dlxnet::error::Code code, dlxnet::StringPiece msg);

  /// Copy the specified status.
  Status(const Status& s);
  Status& operator=(const Status& s);
#ifndef SWIG
  Status(Status&& s) noexcept;
  Status& operator=(Status&& s) noexcept;
#endif  // SWIG

  static Status OK() { return Status(); }

  /// Returns true iff the status indicates success.
  bool ok() const { return (state_ == NULL); }

  dlxnet::error::Code code() const {
    return ok() ? dlxnet::error::OK : state_->code;
  }

  const string& error_message() const {
    return ok() ? empty_string() : state_->msg;
  }

  bool operator==(const Status& x) const;
  bool operator!=(const Status& x) const;

  /// \brief If `ok()`, stores `new_status` into `*this`.  If `!ok()`,
  /// preserves the current status, but may augment with additional
  /// information about `new_status`.
  ///
  /// Convenient way of keeping track of the first error encountered.
  /// Instead of:
  ///   `if (overall_status.ok()) overall_status = new_status`
  /// Use:
  ///   `overall_status.Update(new_status);`
  void Update(const Status& new_status);

  /// \brief Return a string representation of this status suitable for
  /// printing. Returns the string `"OK"` for success.
  string ToString() const;

  // Ignores any errors. This method does nothing except potentially suppress
  // complaints from any tools that are checking that errors are not dropped on
  // the floor.
  void IgnoreError() const;

 private:
  static const string& empty_string();
  struct State {
    dlxnet::error::Code code;
    string msg;
  };
  // OK status has a `NULL` state_.  Otherwise, `state_` points to
  // a `State` structure containing the error code and message(s)
  std::unique_ptr<State> state_;

  void SlowCopyFrom(const State* src);
};

// Helper class to manage multiple child status values.
class StatusGroup {
 public:
  // Utility function to mark a Status as derived. By marking derived status,
  // Derived status messages are ignored when reporting errors to end users.
  static Status MakeDerived(const Status& s);
  static bool IsDerived(const Status& s);

  // Enable warning and error log collection for appending to the aggregated
  // status. This function may be called more than once.
  static void ConfigureLogHistory();

  // Return a merged status with combined child status messages with a summary.
  Status as_summary_status() const;
  // Return a merged status with combined child status messages with
  // concatenation.
  Status as_concatenated_status() const;

  bool ok() const { return ok_; }

  // Augment this group with the child status `status`.
  void Update(const Status& status);

  // Attach recent warning and error log messages
  void AttachLogMessages();
  bool HasLogMessages() const { return !recent_logs_.empty(); }

 private:
  bool ok_ = true;
  size_t num_ok_ = 0;
  std::vector<Status> children_;
  std::vector<std::string> recent_logs_;  // recent warning and error logs
};

inline Status::Status(const Status& s)
    : state_((s.state_ == nullptr) ? nullptr : new State(*s.state_)) {}

inline Status& Status::operator=(const Status& s) {
  // The following condition catches both aliasing (when this == &s),
  // and the common case where both s and *this are ok.
  if (state_ != s.state_) {
    SlowCopyFrom(s.state_.get());
  }
  return *this;
}

#ifndef SWIG
inline Status::Status(Status&& s) noexcept : state_(std::move(s.state_)) {}

inline Status& Status::operator=(Status&& s) noexcept {
  if (state_ != s.state_) {
    state_ = std::move(s.state_);
  }
  return *this;
}
#endif  // SWIG

inline bool Status::operator==(const Status& x) const {
  return (this->state_ == x.state_) || (ToString() == x.ToString());
}

inline bool Status::operator!=(const Status& x) const { return !(*this == x); }

/// @ingroup core
std::ostream& operator<<(std::ostream& os, const Status& x);

typedef std::function<void(const Status&)> StatusCallback;

extern dlxnet::string* TfCheckOpHelperOutOfLine(
    const ::dlxnet::Status& v, const char* msg);

inline dlxnet::string* TfCheckOpHelper(::dlxnet::Status v,
                                           const char* msg) {
  if (v.ok()) return nullptr;
  return TfCheckOpHelperOutOfLine(v, msg);
}

#define TF_DO_CHECK_OK(val, level)                                \
  while (auto _result = ::dlxnet::TfCheckOpHelper(val, #val)) \
  LOG(level) << *(_result)

#define TF_CHECK_OK(val) TF_DO_CHECK_OK(val, FATAL)
#define TF_QCHECK_OK(val) TF_DO_CHECK_OK(val, QFATAL)

// DEBUG only version of TF_CHECK_OK.  Compiler still parses 'val' even in opt
// mode.
#ifndef NDEBUG
#define TF_DCHECK_OK(val) TF_CHECK_OK(val)
#else
#define TF_DCHECK_OK(val) \
  while (false && (::dlxnet::Status::OK() == (val))) LOG(FATAL)
#endif

}  // namespace dlxnet

#endif  // DLXNET_CORE_PLATFORM_STATUS_H_
