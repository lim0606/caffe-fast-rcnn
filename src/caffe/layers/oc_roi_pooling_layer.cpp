// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/fast_rcnn_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void OCROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  OCROIPoolingParameter oc_roi_pool_param = this->layer_param_.oc_roi_pooling_param();
  CHECK_GT(oc_roi_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(oc_roi_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = oc_roi_pool_param.pooled_h();
  pooled_width_ = oc_roi_pool_param.pooled_w();

  CHECK_GE(oc_roi_pool_param.margin_h(), 0)
      << "margin_h must be > 0";
  CHECK_GE(oc_roi_pool_param.margin_w(), 0)
      << "margin_w must be > 0";
  margin_height_ = oc_roi_pool_param.margin_h(); 
  margin_width_ = oc_roi_pool_param.margin_w();

  net_pooled_height_ = pooled_height_ + 2 * margin_height_; 
  net_pooled_width_ = pooled_width_ + 2 * margin_width_; 
  spatial_scale_ = oc_roi_pool_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;

  //printf("ph: %d, pw: %d, mh: %d, mw: %d\n", pooled_height_, pooled_width_, margin_height_, margin_width_);
 
  is_outward_ = margin_height_ > 0 || margin_width_ > 0; 
  
  Reshape(bottom, top); 

  // for buffer to outputs
  const int axis = 2; 
  K_ = buffer_.count(axis); // = buffer_.height() * buffer_.width(); 
  N_inward_ = pooled_height_*pooled_width_; // num_output;
  N_outward_ = (2*margin_width_*pooled_height_ + 2*margin_height_*pooled_width_ + 4*margin_height_*margin_width_); // num_output; 
  M_ = buffer_.count(0, axis); 
 
  inward_multiplier_.Reshape(N_inward_, K_, 1, 1);
  caffe_set(N_inward_*K_, Dtype(0), inward_multiplier_.mutable_cpu_data());

  if (is_outward_) {
    outward_multiplier_.Reshape(N_outward_, K_, 1, 1);
    caffe_set(N_outward_*K_, Dtype(0), outward_multiplier_.mutable_cpu_data()); 
  }
  else { // dummy
    outward_multiplier_.Reshape(1, 1, 1, 1);
    caffe_set(1, Dtype(0), outward_multiplier_.mutable_cpu_data()); 
  }

  Dtype* inward_multiplier_data = inward_multiplier_.mutable_cpu_data();
  Dtype* outward_multiplier_data = outward_multiplier_.mutable_cpu_data();

  int index1 = 0; 
  for (int h=0; h<net_pooled_height_; ++h){
    for (int w=0; w<net_pooled_width_; ++w){
      int index2 = h * net_pooled_width_ + w;
      if (h >= margin_height_ && 
          h < (margin_height_+pooled_height_) &&
          w >= margin_width_ && 
          w < (margin_width_+pooled_width_)) { // inward
        int hh = h - margin_height_; 
        int ww = w - margin_width_; 
        int index3 = hh * pooled_width_ + ww;
        inward_multiplier_data[index3 * K_ + index2] = Dtype(1);
        //printf("h: %d, w: %d // inward, hh: %d, ww: %d\n", h, w, hh, ww);  
      } else { // outward
        outward_multiplier_data[index1 * K_ + index2] = Dtype(1); 
        ++index1; 
        //int hh = h - margin_height_;
        //int ww = w - margin_width_;
        //printf("h: %d, w: %d // outward, hh: %d, ww: %d\n", h, w, hh, ww);  
      }
    }
  }
  /*
  printf("inward----------------\n"); 
  for (int n=0; n<N_inward_; ++n){
    for (int k=0; k<K_; ++k) {
      printf("%.0f ", static_cast<float>(inward_multiplier_data[n * K_ + k])); 
    }
    printf("\n"); 
  }
  printf("\n\n"); 

  index1 = 0;
  for (int h=0; h<net_pooled_height_; ++h){
    for (int w=0; w<net_pooled_width_; ++w){
      int index2 = h * net_pooled_width_ + w;
      if (h >= margin_height_ &&
          h < (margin_height_+pooled_height_) &&
          w >= margin_width_ &&
          w < (margin_width_+pooled_width_)) { // inward
        int hh = h - margin_height_;
        int ww = w - margin_width_;
        int index3 = hh * pooled_width_ + ww;
        printf("%d ", static_cast<int>(inward_multiplier_data[index3 * K_ + index2]));
      } else { // outward
        printf("0 "); //printf("%d ", outward_multiplier_data[index1 * K_ + index2]);
        ++index1;
      }
    }
    printf("\n"); 
  }
  printf("\n\n"); 

  printf("outward----------------\n");
  for (int n=0; n<N_outward_; ++n){
    for (int k=0; k<K_; ++k) {
      printf("%.0f ", static_cast<float>(outward_multiplier_data[n * K_ + k]));
    }
    printf("\n");
  }
  printf("\n\n"); 
  index1 = 0;
  for (int h=0; h<net_pooled_height_; ++h){
    for (int w=0; w<net_pooled_width_; ++w){
      int index2 = h * net_pooled_width_ + w;
      if (h >= margin_height_ &&
          h < (margin_height_+pooled_height_) &&
          w >= margin_width_ &&
          w < (margin_width_+pooled_width_)) { // inward
        int hh = h - margin_height_;
        int ww = w - margin_width_;
        int index3 = hh * pooled_width_ + ww;
        printf("0 "); //printf("%d ", inward_multiplier_data[index3 * K_ + index2]);
      } else { // outward
        printf("%d ", static_cast<int>(outward_multiplier_data[index1 * K_ + index2]));
        ++index1;
      }
    }
    printf("\n"); 
  }
  printf("\n\n"); 
  */
}

template <typename Dtype>
void OCROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  if (is_outward_) {
    top[1]->Reshape(bottom[1]->num(), channels_, 2*margin_width_*pooled_height_ + 2*margin_height_*pooled_width_ + 4*margin_height_*margin_width_, 1);
  }
  else {
    top[1]->Reshape(1, 1, 1, 1); // dummy;
    caffe_set(1, Dtype(0), top[1]->mutable_cpu_data());
  } 

  buffer_.Reshape(bottom[1]->num(), channels_, net_pooled_height_, net_pooled_width_);
  //max_idx_.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_);
  max_idx_.Reshape(bottom[1]->num(), channels_, net_pooled_height_, net_pooled_width_);
}

template <typename Dtype>
void OCROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  //int top_count = top[0]->count();
  int buffer_count = buffer_.count(); 
  //Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* buffer_data = buffer_.mutable_cpu_data(); 
  caffe_set(buffer_count/*top_count*/, Dtype(-FLT_MAX), buffer_data/*top_data*/);
  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(buffer_count/*top_count*/, -1, argmax_data);

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale_);
    int roi_start_h = round(bottom_rois[2] * spatial_scale_);
    int roi_end_w = round(bottom_rois[3] * spatial_scale_);
    int roi_end_h = round(bottom_rois[4] * spatial_scale_);
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    // jhlim
    int roi_height_orig = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width_orig = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = static_cast<int>(static_cast<Dtype>(roi_height_orig) / static_cast<Dtype>(pooled_height_) * static_cast<Dtype>(net_pooled_height_)); 
    int roi_width = static_cast<int>(static_cast<Dtype>(roi_width_orig) / static_cast<Dtype>(pooled_width_) * static_cast<Dtype>(net_pooled_width_));
   
    roi_start_w = roi_start_w + static_cast<Dtype>(roi_width_orig)/2.0 - static_cast<Dtype>(roi_width)/2.0; 
    roi_start_h = roi_start_h + static_cast<Dtype>(roi_height_orig)/2.0 - static_cast<Dtype>(roi_height)/2.0;

    roi_end_w = roi_end_w - static_cast<Dtype>(roi_width_orig)/2.0 + static_cast<Dtype>(roi_width)/2.0;
    roi_end_h = roi_end_h - static_cast<Dtype>(roi_height_orig)/2.0 + static_cast<Dtype>(roi_height)/2.0;

    roi_height = max(roi_end_h - roi_start_h + 1, 1);
    roi_width = max(roi_end_w - roi_start_w + 1, 1);
    // end jhlim
   
    const Dtype bin_size_h = static_cast<Dtype>(roi_height)
                             / static_cast<Dtype>(net_pooled_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width)
                             / static_cast<Dtype>(net_pooled_width_);

    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < net_pooled_height_; ++ph) {
        for (int pw = 0; pw < net_pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / net_pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / net_pooled_height_)
          int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                              * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                              * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                           * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                           * bin_size_w));

          hstart = min(max(hstart + roi_start_h, 0), height_);
          hend = min(max(hend + roi_start_h, 0), height_);
          wstart = min(max(wstart + roi_start_w, 0), width_);
          wend = min(max(wend + roi_start_w, 0), width_);

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pool_index = ph * net_pooled_width_ + pw;
          if (is_empty) {
            buffer_data[pool_index] = 0; //top_data[pool_index] = 0;
            argmax_data[pool_index] = -1;
          }

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width_ + w;
              if (batch_data[index] > buffer_data[pool_index]/*top_data[pool_index]*/) {
                buffer_data[pool_index] = batch_data[index]; //top_data[pool_index] = batch_data[index];
                argmax_data[pool_index] = index;
              }
            }
          }
        }
      }
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      buffer_data/*top_data*/ += buffer_.offset(0, 1); //top[0]->offset(0, 1);
      argmax_data += max_idx_.offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }

  // Split buffer to inward and outward output
  // here
  {
    const Dtype* buffer_data = buffer_.cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const Dtype* inward_multiplier = inward_multiplier_.cpu_data();
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_inward_, K_, (Dtype)1.,
        buffer_data, inward_multiplier, (Dtype)0., top_data);
  }
  if(is_outward_){
    const Dtype* buffer_data = buffer_.cpu_data();
    Dtype* top_data = top[1]->mutable_cpu_data();
    const Dtype* outward_multiplier = outward_multiplier_.cpu_data();
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_outward_, K_, (Dtype)1.,
        buffer_data, outward_multiplier, (Dtype)0., top_data);
  }
}

template <typename Dtype>
void OCROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(OCROIPoolingLayer);
#endif

INSTANTIATE_CLASS(OCROIPoolingLayer);
REGISTER_LAYER_CLASS(OCROIPooling);

}  // namespace caffe
