#include "dlxnet/cc/framework/ops.h"

namespace dlxnet{
    // multi dims input like {{1, 2, 3}, {2, 3, 4}
    Input::Initializer::Initializer(const std::initializer_list<Input::Initializer>& v){
        // when it empty
        if(v.size()<1){
            tensor = Tensor(DT_FLOAT, TensorShape{0});
            return;
        }

        // loop the list and check the same type and same shape at the same time
        auto const& first = *v.begin();
        // Check to make sure that the constituent Initializers are all the same
        // type and same shape.
        for (auto const& e : v) {
            if (e.tensor.dtype() != first.tensor.dtype()) {
                status = errors::InvalidArgument(
                        "Initializer list components should all have the same type");
                return;
            }
            if (!TensorShape{e.tensor.shape()}.IsSameSize(
                        TensorShape{first.tensor.shape()})) {
                status = errors::InvalidArgument(
                        "Initializer list components should all have the same shape");
                return;
            }
        }

        // Form the new shape.
        TensorShape shape{static_cast<int64>(v.size())};
        shape.AppendShape(TensorShape{first.tensor.shape()});

        Tensor t(first.tensor.dtype(), shape);

        // Collate the constituent Tensors.
        size_t offset = 0;
        for (auto const& e : v) {
            Tensor elem = e.tensor;
            if (first.tensor.dtype() == DT_STRING) {
                for (int i = 0; i < elem.NumElements(); ++i) {
                    t.flat<tstring>()(offset + i) = elem.flat<tstring>()(i);
                }
                offset += elem.NumElements();
            } else {
                std::copy_n(elem.tensor_data().data(), elem.TotalBytes(),
                        const_cast<char*>(t.tensor_data().data()) + offset);
                offset += elem.TotalBytes();
            }
        }
        tensor = t;
    }
}
