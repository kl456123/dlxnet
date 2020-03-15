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

#ifndef DLXNET_CORE_LIB_GTL_INLINED_VECTOR_H_
#define DLXNET_CORE_LIB_GTL_INLINED_VECTOR_H_

#include "absl/container/inlined_vector.h"
// TODO(kramerb): This is kept only because lots of targets transitively depend
// on it. Remove all targets' dependencies.
#include "dlxnet/core/platform/macros.h"
#include "dlxnet/core/platform/types.h"

namespace dlxnet {
namespace gtl {

using absl::InlinedVector;

}  // namespace gtl
}  // namespace dlxnet

#endif  // DLXNET_CORE_LIB_GTL_INLINED_VECTOR_H_
