#include "dlxnet/core/util/tensor_format.h"


namespace dlxnet{
    string GetConvnetDataFormatAttrString() {
        return "data_format: { 'NHWC', 'NCHW' }";
      }
}
