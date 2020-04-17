#ifndef DLXNET_STREAM_EXECUTOR_PLATFORM_PORT_H_
#define DLXNET_STREAM_EXECUTOR_PLATFORM_PORT_H_
#include "dlxnet/core/platform/macros.h"
#include "dlxnet/core/platform/types.h"

namespace stream_executor {

using dlxnet::int8;
using dlxnet::int16;
using dlxnet::int32;
using dlxnet::int64;

using dlxnet::uint8;
using dlxnet::uint16;
using dlxnet::uint32;
using dlxnet::uint64;

#if !defined(PLATFORM_GOOGLE)
using std::string;
#endif

#define SE_FALLTHROUGH_INTENDED TF_FALLTHROUGH_INTENDED

}  // namespace stream_executor

#define SE_DISALLOW_COPY_AND_ASSIGN TF_DISALLOW_COPY_AND_ASSIGN
#define SE_MUST_USE_RESULT TF_MUST_USE_RESULT
#define SE_PREDICT_TRUE TF_PREDICT_TRUE
#define SE_PREDICT_FALSE TF_PREDICT_FALSE

#endif
