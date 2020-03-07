#include "dlxnet/core/public/session_options.h"

#include "dlxnet/core/platform/env.h"

namespace dlxnet {

SessionOptions::SessionOptions() : env(Env::Default()) {}

}  // namespace dlxnet
