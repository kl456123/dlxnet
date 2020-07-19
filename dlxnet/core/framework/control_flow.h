#ifndef DLXNET_CORE_FRAMEWORK_CONTROL_FLOW_H_
#define DLXNET_CORE_FRAMEWORK_CONTROL_FLOW_H_
#include "dlxnet/core/platform/types.h"

namespace dlxnet{
    const uint64 kIllegalFrameId = ~0uLL;
    const int64 kIllegalIterId = -1;

    // For the purpose of control flow, every tensor produced by TensorFlow is
    // conceptually tagged by a 'FrameAndIter'. FrameAndIter consists of a
    // 'frame_id' and an 'iter_id'. The tensor value it represents is produced
    // in the frame with frame_id at the iteration of iter_id.
    struct FrameAndIter {
        uint64 frame_id = kIllegalFrameId;
        int64 iter_id = kIllegalIterId;

        FrameAndIter() {}

        FrameAndIter(uint64 frame, int64 iter) {
            frame_id = frame;
            iter_id = iter;
        }

        bool operator==(const FrameAndIter& other) const {
            return (frame_id == other.frame_id && iter_id == other.iter_id);
        }
    };
} // namespace dlxnet



#endif
