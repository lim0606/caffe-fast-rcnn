// Copyright 2013 Yangqing Jia

#include <iostream>  // NOLINT(readability/streams)
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PaddingForward(const int count, const Dtype* in, Dtype* out,
    const int num, const int channel, const int height_in, const int width_in,
    const int pad) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < count) {
    int height_out = height_in + pad + pad;
    int width_out = width_in + pad + pad;
    int w = index % width_in;
    index /= width_in;
    int h = index % height_in;
    index /= height_in;
    int c = index % channel;
    index /= channel;
    out[((index * channel + c) * height_out + h + pad) * width_out + pad + w] =
        in[((index * channel + c) * height_in + h) * width_in + w];
  }
}

template <typename Dtype>
void PaddingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // First, set all data to be zero for the boundary pixels
  CUDA_CHECK(cudaMemset(top_data, 0, sizeof(Dtype) * top[0]->count()));
  // NOLINT_NEXT_LINE(whitespace/operators)
  PaddingForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, NUM_, CHANNEL_, HEIGHT_IN_, WIDTH_IN_,
      PAD_);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void PaddingBackward(const int count, const Dtype* in, Dtype* out,
    const int num, const int channel, const int height_in, const int width_in,
    const int pad) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < count) {
    int height_out = height_in + pad + pad;
    int width_out = width_in + pad + pad;
    int w = index % width_in;
    index /= width_in;
    int h = index % height_in;
    index /= height_in;
    int c = index % channel;
    index /= channel;
    out[((index * channel + c) * height_in + h) * width_in + w] =
        in[((index * channel + c) * height_out + h + pad) *
           width_out + pad + w];
  }
}

template <typename Dtype>
void PaddingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, //const bool propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    PaddingBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_diff, NUM_, CHANNEL_, HEIGHT_IN_, WIDTH_IN_,
        PAD_);
    CUDA_POST_KERNEL_CHECK;

    //const Dtype* top_diff = top[0]->gpu_diff(); 
    //Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    //caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
    //for (int n = 0; n < NUM_; ++n) {
    //  for (int c = 0; c < CHANNEL_; ++c) {
    //    for (int h = 0; h < HEIGHT_IN_; ++h) {
    //      // copy the width part
    //      caffe_gpu_axpy(WIDTH_IN_, (Dtype)1.,
    //         top_diff+top[0]->offset(n, c, h + PAD_, PAD_),
    //         bottom_diff+bottom[0]->offset(n, c, h));
    //    }
    //  }
    //}
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PaddingLayer);

}  // namespace caffe
