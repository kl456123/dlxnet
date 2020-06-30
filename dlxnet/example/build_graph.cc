/////
//demostrate how to build graph using some operations
// c++ version example

#include <memory>

#include "dlxnet/core/public/session.h"
#include "dlxnet/cc/ops/const_op.h"
#include "dlxnet/cc/ops/array_ops.h"

using dlxnet::string;
using dlxnet::Status;
using dlxnet::Scope;
using dlxnet::Tensor;

Status TestMatMul(){
    using namespace dlxnet::ops;
    Scope root = Scope::NewRootScope();
    auto a = Const(root, {{1, 2}, {2, 4}});

    auto b = Const(root, {{2, 2}, {1, 1}});
    // axb
    string output_name = "MatMul_1";
    auto m = MatMul(root, a, b);

    auto m2 = MatMul(root, m, b);

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensor.
    dlxnet::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    auto options = dlxnet::SessionOptions();
    options.config.mutable_gpu_options()->set_visible_device_list("0");

    std::unique_ptr<dlxnet::Session> session(
            dlxnet::NewSession(options));
    TF_RETURN_IF_ERROR(session->Create(graph));

    std::vector<Tensor> out_tensors;
    TF_RETURN_IF_ERROR(session->Run({}, {output_name}, {}, &out_tensors));
    std::cout<<out_tensors[0].DebugString()<<std::endl;
    return Status::OK();
}

int main(){
    Status status = TestMatMul();
    if(!status.ok()){
        LOG(ERROR) << "Test MatMul model failed: " << status;
        return -1;
    }
}
