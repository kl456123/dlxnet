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

// A minimal but useful C++ example showing how to load an Imagenet-style object
// recognition TensorFlow model, prepare input images for it, run them through
// the graph, and interpret the results.
//
// It's designed to have as few dependencies and be as clear as possible, so
// it's more verbose than it could be in production code. In particular, using
// auto for the types of a lot of the returned values from TensorFlow calls can
// remove a lot of boilerplate, but I find the explicit types useful in sample
// code to make it simple to look up the classes involved.
//
// To use it, compile and then run in a working directory with the
// learning/brain/tutorials/label_image/data/ folder below it, and you should
// see the top five labels for the example Lena image output. You can then
// customize it to use your own models or images by changing the file names at
// the top of the main() function.
//
// The googlenet_graph.pb file included by default is created from Inception.
//
// Note that, for GIF inputs, to reuse existing code, only single-frame ones
// are supported.

#include <fstream>
#include <utility>
#include <vector>

#include "dlxnet/core/framework/graph.pb.h"
#include "dlxnet/core/framework/tensor.h"
// #include "dlxnet/core/graph/default_device.h"
// #include "dlxnet/core/graph/graph_def_builder.h"
#include "dlxnet/core/lib/core/errors.h"
#include "dlxnet/core/lib/core/stringpiece.h"
// #include "dlxnet/core/lib/core/threadpool.h"
#include "dlxnet/core/lib/io/path.h"
// #include "dlxnet/core/lib/strings/str_util.h"
// #include "dlxnet/core/lib/strings/stringprintf.h"
#include "dlxnet/core/platform/env.h"
#include "dlxnet/core/platform/init_main.h"
#include "dlxnet/core/platform/logging.h"
#include "dlxnet/core/platform/types.h"
#include "dlxnet/core/public/session.h"
#include "dlxnet/core/util/command_line_flags.h"

// These are all common classes it's handy to reference with no namespace.
using dlxnet::Flag;
using dlxnet::int32;
using dlxnet::Status;
using dlxnet::string;
using dlxnet::Tensor;

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
Status ReadLabelsFile(const string& file_name, std::vector<string>* result,
                      size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    return dlxnet::errors::NotFound("Labels file ", file_name,
                                        " not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return Status::OK();
}

static Status ReadEntireFile(dlxnet::Env* env, const string& filename,
                             Tensor* output) {
  dlxnet::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

  string contents;
  contents.resize(file_size);

  std::unique_ptr<dlxnet::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  dlxnet::StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return dlxnet::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
  }
  // output->scalar<string>()() = string(data);
  return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<dlxnet::Session>* session) {
  dlxnet::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(dlxnet::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return dlxnet::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(dlxnet::NewSession(dlxnet::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
// Status GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
                    // Tensor* indices, Tensor* scores) {
  // auto root = tensorflow::Scope::NewRootScope();
  // using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  // string output_name = "top_k";
  // TopK(root.WithOpName(output_name), outputs[0], how_many_labels);
  // // This runs the GraphDef network definition that we've just constructed, and
  // // returns the results in the output tensors.
  // tensorflow::GraphDef graph;
  // TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  // std::unique_ptr<tensorflow::Session> session(
      // tensorflow::NewSession(tensorflow::SessionOptions()));
  // TF_RETURN_IF_ERROR(session->Create(graph));
  // // The TopK node returns two outputs, the scores and their original indices,
  // // so we have to append :0 and :1 to specify them both.
  // std::vector<Tensor> out_tensors;
  // TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                  // {}, &out_tensors));
  // *scores = out_tensors[0];
  // *indices = out_tensors[1];
  // return Status::OK();
// }

// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
// Status PrintTopLabels(const std::vector<Tensor>& outputs,
                      // const string& labels_file_name) {
  // std::vector<string> labels;
  // size_t label_count;
  // Status read_labels_status =
      // ReadLabelsFile(labels_file_name, &labels, &label_count);
  // if (!read_labels_status.ok()) {
    // LOG(ERROR) << read_labels_status;
    // return read_labels_status;
  // }
  // const int how_many_labels = std::min(5, static_cast<int>(label_count));
  // Tensor indices;
  // Tensor scores;
  // TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  // tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
  // tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  // for (int pos = 0; pos < how_many_labels; ++pos) {
    // const int label_index = indices_flat(pos);
    // const float score = scores_flat(pos);
    // LOG(INFO) << labels[label_index] << " (" << label_index << "): " << score;
  // }
  // return Status::OK();
// }

// This is a testing function that returns whether the top label index is the
// one that's expected.
// Status CheckTopLabel(const std::vector<Tensor>& outputs, int expected,
                     // bool* is_expected) {
  // *is_expected = false;
  // Tensor indices;
  // Tensor scores;
  // const int how_many_labels = 1;
  // TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  // tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  // if (indices_flat(0) != expected) {
    // LOG(ERROR) << "Expected label #" << expected << " but got #"
               // << indices_flat(0);
    // *is_expected = false;
  // } else {
    // *is_expected = true;
  // }
  // return Status::OK();
// }

int main(int argc, char* argv[]) {
  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than inception_v3, then you'll need to update these.
  string image = "tensorflow/examples/label_image/data/grace_hopper.jpg";
  string graph =
      "tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb";
  string labels =
      "tensorflow/examples/label_image/data/imagenet_slim_labels.txt";
  int32 input_width = 299;
  int32 input_height = 299;
  float input_mean = 0;
  float input_std = 255;
  string input_layer = "input";
  string output_layer = "InceptionV3/Predictions/Reshape_1";
  bool self_test = false;
  string root_dir = "";
  std::vector<Flag> flag_list = {
      Flag("image", &image, "image to be processed"),
      Flag("graph", &graph, "graph to be executed"),
      Flag("labels", &labels, "name of file containing labels"),
      Flag("input_width", &input_width, "resize image to this width in pixels"),
      Flag("input_height", &input_height,
           "resize image to this height in pixels"),
      Flag("input_mean", &input_mean, "scale pixel values to this mean"),
      Flag("input_std", &input_std, "scale pixel values to this std deviation"),
      Flag("input_layer", &input_layer, "name of input layer"),
      Flag("output_layer", &output_layer, "name of output layer"),
      Flag("self_test", &self_test, "run a self test"),
      Flag("root_dir", &root_dir,
           "interpret image and graph file names relative to this directory"),
  };
  string usage = dlxnet::Flags::Usage(argv[0], flag_list);
  const bool parse_result = dlxnet::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  dlxnet::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  // First we load and initialize the model.
  std::unique_ptr<dlxnet::Session> session;
  string graph_path = dlxnet::io::JoinPath(root_dir, graph);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  std::vector<Tensor> resized_tensors;
  string image_path = dlxnet::io::JoinPath(root_dir, image);
  // Status read_tensor_status =
      // ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
                              // input_std, &resized_tensors);
  // if (!read_tensor_status.ok()) {
    // LOG(ERROR) << read_tensor_status;
    // return -1;
  // }
  const Tensor& resized_tensor = resized_tensors[0];

  // Actually run the image through the model.
  std::vector<Tensor> outputs;
  Status run_status = session->Run({{input_layer, resized_tensor}},
                                   {output_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }

  // This is for automated testing to make sure we get the expected result with
  // the default settings. We know that label 653 (military uniform) should be
  // the top label for the Admiral Hopper image.
  // if (self_test) {
    // bool expected_matches;
    // Status check_status = CheckTopLabel(outputs, 653, &expected_matches);
    // if (!check_status.ok()) {
      // LOG(ERROR) << "Running check failed: " << check_status;
      // return -1;
    // }
    // if (!expected_matches) {
      // LOG(ERROR) << "Self-test failed!";
      // return -1;
    // }
  // }

  // Do something interesting with the results we've generated.
  // Status print_status = PrintTopLabels(outputs, labels);
  // if (!print_status.ok()) {
    // LOG(ERROR) << "Running print failed: " << print_status;
    // return -1;
  // }

  return 0;
}
