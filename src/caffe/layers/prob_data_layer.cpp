#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <sstream> 

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ProbDataLayer<Dtype>::~ProbDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype> 
vector<Dtype> ProbDataLayer<Dtype>::ReadProbs(int line_id) {
  float* prob_data = data_.mutable_cpu_data();
  vector<Dtype> probs; probs.resize(0); 

  for (int i = 0; i < num_classes_; ++i) {
    probs.push_back(static_cast<Dtype>(prob_data[line_id*num_classes_+i])); 
  } 

  return probs; 
}

template <typename Dtype>
void ProbDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const string& source = this->layer_param_.prob_data_param().source();
  LOG(INFO) << "Opening file " << source;
 
  caffe::BlobProtoVector blob_proto_vec;
  blob_proto_vec.Clear();
  ReadProtoFromBinaryFile(source, &blob_proto_vec);

  const caffe::BlobProto& blob_proto_prob = blob_proto_vec.blobs(0);
  const caffe::BlobProto& blob_proto_label = blob_proto_vec.blobs(1);

  data_.Reshape(1, 1, 1, 1);  
  label_.Reshape(1, 1, 1, 1);

  data_.FromProto(blob_proto_prob, true);
  label_.FromProto(blob_proto_label, true);

  //float* prob_data = data_.mutable_cpu_data();
  //float* label_data = label_.mutable_cpu_data();

  num_data_ = data_.shape(0);
  num_classes_ = data_.shape(1); 

  LOG(INFO) << "num_data: " << num_data_;
  LOG(INFO) << "num_classes: " << num_classes_;

  //for (int i = 0; i < 1/*numdata*/; ++i) {
  //  printf("%e", prob_data[i*num_classes_+0]);
  //  for (int j = 1; j < num_classes_; ++j) {
  //    printf(",%e", prob_data[i*num_classes_+j]);
  //  }
  //  printf("\n");
  //}
 
  //std::ifstream infile(source.c_str());
  //string line;
  //int label;
  //while (infile >> line >> label) {
  //  lines_.push_back(std::make_pair(line, label));
  //}

  //LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  
  // Read an probs, and use it to initialize the top blob.
  vector<int> top_shape(2);
  top_shape[0] = 1; 
  top_shape[1] = num_classes_;
  vector<Dtype> probs = ReadProbs(lines_id_);  
  CHECK_EQ(probs.size(), num_classes_); 
  this->transformed_data_.Reshape(top_shape);

  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.prob_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels();

  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

//template <typename Dtype>
//void ProbDataLayer<Dtype>::ShuffleProbs() {
//  caffe::rng_t* prefetch_rng =
//      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
//  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
//}

// This function is called on prefetch thread
template <typename Dtype>
void ProbDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ProbDataParameter prob_data_param = this->layer_param_.prob_data_param();
  const int batch_size = prob_data_param.batch_size();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  vector<int> top_shape(2);
  top_shape[0] = 1;
  top_shape[1] = num_classes_;
  vector<Dtype> probs = ReadProbs(lines_id_);
  CHECK_EQ(probs.size(), num_classes_);
  this->transformed_data_.Reshape(top_shape);

  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  float* label_data = label_.mutable_cpu_data();

  // datum scales
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(num_data_, lines_id_);
    vector<Dtype> probs = ReadProbs(lines_id_);
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    Dtype* transformed_data = this->transformed_data_.mutable_cpu_data();
    for (int i = 0; i < num_classes_; ++i) {
      transformed_data[i] = probs[i]; 
    } 
    trans_time += timer.MicroSeconds();

    prefetch_label[item_id] = static_cast<int>(label_data[lines_id_]);
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= num_data_) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
//      if (this->layer_param_.prob_data_param().shuffle()) {
//        ShuffleProbs();
//      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ProbDataLayer);
REGISTER_LAYER_CLASS(ProbData);

}  // namespace caffe
