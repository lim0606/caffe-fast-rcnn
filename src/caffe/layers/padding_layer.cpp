// Copyright 2013 Yangqing Jia

#include <iostream>  // NOLINT(readability/streams)
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void PaddingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) {
  PaddingParameter padding_param = this->layer_param().padding_param();
  PAD_ = padding_param.pad();
  Reshape(bottom, top); 
}

template <typename Dtype>
void PaddingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "Padding Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "Padding Layer takes a single blob as output.";
  NUM_ = bottom[0]->num();
  CHANNEL_ = bottom[0]->channels();
  HEIGHT_IN_ = bottom[0]->height();
  WIDTH_IN_ = bottom[0]->width();
  HEIGHT_OUT_ = HEIGHT_IN_ + PAD_ * 2;
  WIDTH_OUT_ = WIDTH_IN_ + PAD_ * 2;
  top[0]->Reshape(NUM_, CHANNEL_, HEIGHT_OUT_, WIDTH_OUT_);
}

template <typename Dtype>
void PaddingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  memset(top_data, 0, sizeof(Dtype) * top[0]->count());
  // In short, top[n, c, h, w] = bottom[n, c, h-pad, w-pad] if in range
  for (int n = 0; n < NUM_; ++n) {
    for (int c = 0; c < CHANNEL_; ++c) {
      for (int h = 0; h < HEIGHT_IN_; ++h) {
        // copy the width part
        memcpy(
            top_data + ((n * CHANNEL_ + c) * HEIGHT_OUT_ + h + PAD_)
                * WIDTH_OUT_ + PAD_,
            bottom_data + ((n * CHANNEL_ + c) * HEIGHT_IN_ + h) * WIDTH_IN_,
            sizeof(Dtype) * WIDTH_IN_);
      }
    }
  }
}

template <typename Dtype>
void PaddingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, //const bool propagate_down, 
      const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int n = 0; n < NUM_; ++n) {
      for (int c = 0; c < CHANNEL_; ++c) {
        for (int h = 0; h < HEIGHT_IN_; ++h) {
          // copy the width part
          memcpy(
              bottom_diff + ((n * CHANNEL_ + c) * HEIGHT_IN_ + h) * WIDTH_IN_,
              top_diff + ((n * CHANNEL_ + c) * HEIGHT_OUT_ + h + PAD_)
                  * WIDTH_OUT_ + PAD_,
              sizeof(Dtype) * WIDTH_IN_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(PaddingLayer);
#endif

INSTANTIATE_CLASS(PaddingLayer);
REGISTER_LAYER_CLASS(Padding);

}  // namespace caffe
