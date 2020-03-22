#ifndef DLXNET_CORE_COMMON_RUNTIME_PLACER_H_
#define DLXNET_CORE_COMMON_RUNTIME_PLACER_H_
#include "dlxnet/core/lib/status.h"


namespace dlxnet{

    class Placer{
        public:
            Placer(){}
            Status Run(){return Status::OK();}
    };
}


#endif
