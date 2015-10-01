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

namespace caffe {

template <typename Dtype>
__global__ void OCROIPoolForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, 
    const int channels, const int height, const int width, 
    const int pooled_height, const int pooled_width, 
    const int net_pooled_height, const int net_pooled_width, 
    const Dtype* bottom_rois, Dtype* top_data, int* argmax_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % net_pooled_width;
    int ph = (index / net_pooled_width) % net_pooled_height;
    int c = (index / net_pooled_width / net_pooled_height) % channels;
    int n = index / net_pooled_width / net_pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale);
    int roi_start_h = round(bottom_rois[2] * spatial_scale);
    int roi_end_w = round(bottom_rois[3] * spatial_scale);
    int roi_end_h = round(bottom_rois[4] * spatial_scale);

    // jhlim
    int roi_height_orig = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width_orig = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = static_cast<int>(static_cast<Dtype>(roi_height_orig) / static_cast<Dtype>(pooled_height) * static_cast<Dtype>(net_pooled_height));
    int roi_width = static_cast<int>(static_cast<Dtype>(roi_width_orig) / static_cast<Dtype>(pooled_width) * static_cast<Dtype>(net_pooled_width));

    roi_start_w = roi_start_w + static_cast<Dtype>(roi_width_orig)/2.0 - static_cast<Dtype>(roi_width)/2.0;
    roi_start_h = roi_start_h + static_cast<Dtype>(roi_height_orig)/2.0 - static_cast<Dtype>(roi_height)/2.0;
    roi_end_w = roi_end_w - static_cast<Dtype>(roi_width_orig)/2.0 + static_cast<Dtype>(roi_width)/2.0;
    roi_end_h = roi_end_h - static_cast<Dtype>(roi_height_orig)/2.0 + static_cast<Dtype>(roi_height)/2.0;

    // Force malformed ROIs to be 1x1
    roi_height = max(roi_end_h - roi_start_h + 1, 1);
    roi_width = max(roi_end_w - roi_start_w + 1, 1);
    // end jhlim


    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(net_pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(net_pooled_width);

    int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        if (bottom_data[bottom_index] > maxval) {
          maxval = bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

template <typename Dtype>
void OCROIPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  //Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* buffer_data = buffer_.mutable_gpu_data(); 
  int* argmax_data = max_idx_.mutable_gpu_data();
  int count = buffer_.count();//top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  OCROIPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, net_pooled_height_, net_pooled_width_, bottom_rois, buffer_data/*top_data*/, argmax_data);
  CUDA_POST_KERNEL_CHECK;

  /*printf("bottom----------------------------------------\n"); 
  for (int n=0; n<bottom[0]->num(); ++n) {
    for (int c=0; c<bottom[0]->channels(); ++c) {
      printf("n: %d, c: %d\n", n, c); 
      for (int h=0; h<bottom[0]->height(); ++h) {
        for (int w=0; w<bottom[0]->width(); ++w) {
          const Dtype* bottom_data = bottom[0]->cpu_data();
          int index = n * (channels_ * net_pooled_height_ * net_pooled_width_) + c * (net_pooled_height_ * net_pooled_width_) + h * net_pooled_width_ + w; 
          printf("%.2f ", static_cast<float>(bottom_data[index])); 
        }
        printf("\n");
      }
      printf("\n\n");  
    }
  }*/

  // Split buffer to inward and outward output
  // here
  {
    const Dtype* buffer_data = buffer_.gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* inward_multiplier = inward_multiplier_.gpu_data();
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_inward_, K_, (Dtype)1.,
        buffer_data, inward_multiplier, (Dtype)0., top_data);
  }
  if(is_outward_){
    const Dtype* buffer_data = buffer_.gpu_data();
    Dtype* top_data = top[1]->mutable_gpu_data();
    const Dtype* outward_multiplier = outward_multiplier_.gpu_data();
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_outward_, K_, (Dtype)1.,
        buffer_data, outward_multiplier, (Dtype)0., top_data);
  }

  /*
  for (int n=0; n<buffer_.num(); ++n) {
    for (int c=0; c<buffer_.channels(); ++c) {
      int index1 = 0;
      for (int h=0; h<buffer_.height(); ++h) {
        for (int w=0; w<buffer_.width(); ++w) {
          if (h >= margin_height_ &&
              h < (margin_height_+pooled_height_) &&
              w >= margin_width_ &&
              w < (margin_width_+pooled_width_)) { // inward
            int hh = h - margin_height_;
            int ww = w - margin_width_;
            const Dtype* buffer_data = buffer_.cpu_data();
            Dtype* top_data = top[0]->mutable_cpu_data();
            top_data[n * (channels_ * pooled_height_ * pooled_width_) + c * (pooled_height_ * pooled_width_) + hh * pooled_width_ + ww] = buffer_data[n * (channels_ * net_pooled_height_ * net_pooled_width_) + c * (net_pooled_height_ * net_pooled_width_) + h * net_pooled_width_ + w];
          } else { // outward
            const Dtype* buffer_data = buffer_.cpu_data();
            Dtype* top_data = top[1]->mutable_cpu_data();
            top_data[n * (channels_ * top[1]->height()) + c * top[1]->height() + index1] = buffer_data[n * (channels_ * net_pooled_height_ * net_pooled_width_) + c * (net_pooled_height_ * net_pooled_width_) + h * net_pooled_width_ + w];
            ++index1;
          }
        }
      }
      CHECK_EQ(index1, top[1]->height()) << "WTF"; 
    }
  }
  */
  
  /*printf("buffer---------------------------------------\n");  
  for (int n=0; n<buffer_.num(); ++n) {
    for (int c=0; c<buffer_.channels(); ++c) {
      printf("n: %d, c: %d\n", n, c);
      for (int h=0; h<buffer_.height(); ++h) {
        for (int w=0; w<buffer_.width(); ++w) {
          const Dtype* buffer_data = buffer_.cpu_data();
          int index = n * (channels_ * net_pooled_height_ * net_pooled_width_) + c * (net_pooled_height_ * net_pooled_width_) + h * net_pooled_width_ + w;
          printf("%.2f ", static_cast<float>(buffer_data[index]));
        }
        printf("\n");
      }
      printf("\n\n");
    }
  }

  printf("top[0]---------------------------------------\n"); 
  for (int n=0; n<top[0]->num(); ++n) {
    for (int c=0; c<top[0]->channels(); ++c) {
      printf("n: %d, c: %d\n", n, c);
      for (int h=0; h<top[0]->height(); ++h) {
        for (int w=0; w<top[0]->width(); ++w) {
          const Dtype* top_data = top[0]->cpu_data();
          int index = n * (channels_ * pooled_height_ * pooled_width_) + c * (pooled_height_ * pooled_width_) + h * pooled_width_ + w;
          printf("%.2f ", static_cast<float>(top_data[index]));
        }
        printf("\n");
      }
      printf("\n\n");
    }
  }
 

  printf("top[1]---------------------------------------\n");
  for (int n=0; n<buffer_.num(); ++n) {
    for (int c=0; c<buffer_.channels(); ++c) {
      printf("n: %d, c: %d\n", n, c);
      int index1 = 0;
      for (int h=0; h<buffer_.height(); ++h) {
        for (int w=0; w<buffer_.width(); ++w) {
          if (h >= margin_height_ &&
              h < (margin_height_+pooled_height_) &&
              w >= margin_width_ &&
              w < (margin_width_+pooled_width_)) { // inward
            printf("%.2f ", -1.0);
          } else { // outward
            const Dtype* top_data = top[1]->cpu_data();
            printf("%.2f ", top_data[n * (channels_ * top[1]->height()) + c * top[1]->height() + index1]); 
            ++index1;
          }
        }
        printf("\n");
      }
      printf("\n\n");
    }
  }*/
}

template <typename Dtype>
__global__ void OCROIPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, 
    const int net_pooled_height, const int net_pooled_width, 
    Dtype* bottom_diff,
    const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
      int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
      int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
      int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

      // jhlim
      int roi_height_orig = max(roi_end_h - roi_start_h + 1, 1);
      int roi_width_orig = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = static_cast<int>(static_cast<Dtype>(roi_height_orig) / static_cast<Dtype>(pooled_height) * static_cast<Dtype>(net_pooled_height));
      int roi_width = static_cast<int>(static_cast<Dtype>(roi_width_orig) / static_cast<Dtype>(pooled_width) * static_cast<Dtype>(net_pooled_width));
  
      roi_start_w = roi_start_w + static_cast<Dtype>(roi_width_orig)/2.0 - static_cast<Dtype>(roi_width)/2.0;                                                                           roi_start_h = roi_start_h + static_cast<Dtype>(roi_height_orig)/2.0 - static_cast<Dtype>(roi_height)/2.0;
  
      roi_end_w = roi_end_w - static_cast<Dtype>(roi_width_orig)/2.0 + static_cast<Dtype>(roi_width)/2.0;                                                                               roi_end_h = roi_end_h - static_cast<Dtype>(roi_height_orig)/2.0 + static_cast<Dtype>(roi_height)/2.0;
  
      roi_height = max(roi_end_h - roi_start_h + 1, 1);
      roi_width = max(roi_end_w - roi_start_w + 1, 1); 
      // end jhlim


      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * net_pooled_height * net_pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const int* offset_argmax_data = argmax_data + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      //// Force malformed ROIs to be 1x1
      //int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      //int roi_height = max(roi_end_h - roi_start_h + 1, 1);

      Dtype bin_size_h = static_cast<Dtype>(roi_height)
                         / static_cast<Dtype>(net_pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width)
                         / static_cast<Dtype>(net_pooled_width);

      int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
      int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
      int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

      phstart = min(max(phstart, 0), net_pooled_height);
      phend = min(max(phend, 0), net_pooled_height);
      pwstart = min(max(pwstart, 0), net_pooled_width);
      pwend = min(max(pwend, 0), net_pooled_width);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (offset_argmax_data[ph * net_pooled_width + pw] == (h * width + w)) {
            gradient += offset_top_diff[ph * net_pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void OCROIPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  // Concate diff from tops to buffers 
  // here
  {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_inward_, (Dtype)1.,
        top_diff, inward_multiplier_.gpu_data(), (Dtype)0.,
        buffer_.mutable_gpu_diff());
  }
  if (is_outward_) {
    const Dtype* top_diff = top[1]->gpu_diff();
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_outward_, (Dtype)1.,
        top_diff, outward_multiplier_.gpu_data(), (Dtype)1.,
        buffer_.mutable_gpu_diff());
  }

  const Dtype* bottom_rois = bottom[1]->gpu_data();
  //const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* buffer_diff = buffer_.gpu_diff(); 
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int* argmax_data = max_idx_.gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  OCROIPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, buffer_diff/*top_diff*/, argmax_data, buffer_.num()/*top[0]->num()*/, spatial_scale_, channels_,
      height_, width_, pooled_height_, pooled_width_, net_pooled_height_, net_pooled_width_, bottom_diff, bottom_rois);
  CUDA_POST_KERNEL_CHECK;

  /*
  printf("bottom diff----------------------------------------\n");
  for (int n=0; n<bottom[0]->num(); ++n) {
    for (int c=0; c<bottom[0]->channels(); ++c) {
      printf("n: %d, c: %d\n", n, c);
      for (int h=0; h<bottom[0]->height(); ++h) {
        for (int w=0; w<bottom[0]->width(); ++w) {
          const Dtype* bottom_diff = bottom[0]->cpu_diff();
          int index = n * (channels_ * net_pooled_height_ * net_pooled_width_) + c * (net_pooled_height_ * net_pooled_width_) + h * net_pooled_width_ + w;
          printf("%.2f ", static_cast<float>(bottom_diff[index]));
        }
        printf("\n");
      }
      printf("\n\n");
    }
  }

  printf("argmax_data----------------------------------------\n");
  for (int n=0; n<max_idx_.num(); ++n) {
    for (int c=0; c<max_idx_.channels(); ++c) {
      printf("n: %d, c: %d\n", n, c);
      for (int h=0; h<max_idx_.height(); ++h) {
        for (int w=0; w<max_idx_.width(); ++w) {
          const int* argmax_data = max_idx_.cpu_data();
          int index = n * (channels_ * net_pooled_height_ * net_pooled_width_) + c * (net_pooled_height_ * net_pooled_width_) + h * net_pooled_width_ + w;
          printf("%d ", argmax_data[index]);
        }
        printf("\n");
      }
      printf("\n\n");
    }
  }

  printf("buffer diff---------------------------------------\n");  
  for (int n=0; n<buffer_.num(); ++n) {
    for (int c=0; c<buffer_.channels(); ++c) {
      printf("n: %d, c: %d\n", n, c);
      for (int h=0; h<buffer_.height(); ++h) {
        for (int w=0; w<buffer_.width(); ++w) {
          const Dtype* buffer_diff = buffer_.cpu_diff();
          int index = n * (channels_ * net_pooled_height_ * net_pooled_width_) + c * (net_pooled_height_ * net_pooled_width_) + h * net_pooled_width_ + w;
          printf("%.2f ", static_cast<float>(buffer_diff[index]));
        }
        printf("\n");
      }
      printf("\n\n");
    }
  }

  printf("top[0] diff---------------------------------------\n"); 
  for (int n=0; n<top[0]->num(); ++n) {
    for (int c=0; c<top[0]->channels(); ++c) {
      printf("n: %d, c: %d\n", n, c);
      for (int h=0; h<top[0]->height(); ++h) {
        for (int w=0; w<top[0]->width(); ++w) {
          const Dtype* top_diff = top[0]->cpu_diff();
          int index = n * (channels_ * pooled_height_ * pooled_width_) + c * (pooled_height_ * pooled_width_) + h * pooled_width_ + w;
          printf("%.2f ", static_cast<float>(top_diff[index]));
        }
        printf("\n");
      }
      printf("\n\n");
    }
  }
 

  printf("top[1] diff---------------------------------------\n");
  for (int n=0; n<buffer_.num(); ++n) {
    for (int c=0; c<buffer_.channels(); ++c) {
      printf("n: %d, c: %d\n", n, c);
      int index1 = 0;
      for (int h=0; h<buffer_.height(); ++h) {
        for (int w=0; w<buffer_.width(); ++w) {
          if (h >= margin_height_ &&
              h < (margin_height_+pooled_height_) &&
              w >= margin_width_ &&
              w < (margin_width_+pooled_width_)) { // inward
            printf("%.2f ", -1.0);
          } else { // outward
            const Dtype* top_diff = top[1]->cpu_diff();
            printf("%.2f ", top_diff[n * (channels_ * top[1]->height()) + c * top[1]->height() + index1]); 
            ++index1;
          }
        }
        printf("\n");
      }
      printf("\n\n");
    }
  }
  */  
}

INSTANTIATE_LAYER_GPU_FUNCS(OCROIPoolingLayer);

}  // namespace caffe
