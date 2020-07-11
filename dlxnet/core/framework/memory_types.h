#ifndef DLXNET_CORE_FRAMEWORK_MEMORY_TYPES_H_
#define DLXNET_CORE_FRAMEWORK_MEMORY_TYPES_H_
#include "dlxnet/core/framework/op.h"
#include "dlxnet/core/framework/types.h"

namespace dlxnet{
    class NodeDef;

  // Returns into *{input,output}_memory_types the memory type of each
  // {input,output} tensor.
  //
  // REQUIRES: * '*_memory_types' is not nullptr.
  //           * def has all attrs specified (e.g. using AddDefaultsToNodeDef()).
  Status MemoryTypesForNode(const OpRegistryInterface* op_registry,
                            const DeviceType& device_type, const NodeDef& ndef,
                            MemoryTypeVector* input_memory_types,
                            MemoryTypeVector* output_memory_types);
} // namespace dlxnet



#endif
