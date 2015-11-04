/**
 * 3D Grid layer written by Jaehyun Lim, modified from https://github.com/BVLC/caffe/pull/2033/files
 *
 * TO DO: 
 * 1. propagate direction
 * 2. handling different batchsizes between TRAIN and TEST 
 *    - current implementation specify the blob shape of input(s) ("x" or "c" and "h") once when LayerSetUp is called. Because of that, the system provide error when batchsizes in TRAIN and TEST are different 
      - message as follows
        F1104 10:56:49.998088 29439 net.cpp:788] Check failed: target_blobs[j]->shape() == source_blob->shape() Cannot share param 0 weights from layer '3dgridlstm'; shape mismatch.  Source param shape is <batch_in_train> <num_output> (6400); target param shape is <batch_in_test> <num_output> (10000)
 * 3. test code
 */

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/grid_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ThreeDGridLSTMLayer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
  //names->resize(2);
  //(*names)[0] = "h_0";
  //(*names)[1] = "c_0";

  names->resize( (this->height_ + this->width_) * 2 ); 

  // Initial inputs of width increment direction (i.e. w0)
  for (int i = 0; i < this->height_; ++i) {
    (*names)[i*2]   = "h"+this->int_to_str(i+1)+"_w0_d1/w/h";
    (*names)[i*2+1] = "h"+this->int_to_str(i+1)+"_w0_d1/w/c";
  }

  // Initial inputs of height increment direction (i.e. h0)
  for (int i = 0; i < this->width_; ++i) {
    (*names)[this->height_*2+i*2]   = "h0_w"+this->int_to_str(i+1)+"_d1/h/h";
    (*names)[this->height_*2+i*2+1] = "h0_w"+this->int_to_str(i+1)+"_d1/h/c";
  }
}

//template <typename Dtype>
//void ThreeDGridLSTMLayer<Dtype>::RecurrentOutputBlobNames(vector<string>* names) const {
//  names->resize(2);
//  (*names)[0] = "h_" + this->int_to_str(this->T_);
//  (*names)[1] = "c_T";
//}

template <typename Dtype>
void ThreeDGridLSTMLayer<Dtype>::RecurrentInputShapes(vector<BlobShape>* shapes) const {
  const int num_output = this->layer_param_.grid_param().num_output();
  //const int num_blobs = 2;
  const int num_blobs = (this->height_ + this->width_) * 2;
  shapes->resize(num_blobs);
  for (int i = 0; i < num_blobs; ++i) {
    (*shapes)[i].Clear();
    (*shapes)[i].add_dim(this->num_);  // batch size
    (*shapes)[i].add_dim(num_output);
    //(*shapes)[i].add_dim(1);
    //(*shapes)[i].add_dim(1);
  }
}

template <typename Dtype>
void ThreeDGridLSTMLayer<Dtype>::InputBlobNames(vector<string>* names, const int num_bottom) const {
  CHECK_LE(num_bottom, 2) << "num input must be less than equal to 2"; 
  names->resize(num_bottom);
  if (num_bottom == 1) {
    (*names)[0] = "x";
  } 
  else { // num_bottom == 2
    (*names)[0] = "d0/d/h";
    (*names)[1] = "d0/d/c";
  }
}

template <typename Dtype>
void ThreeDGridLSTMLayer<Dtype>::OutputBlobNames(vector<string>* names) const {
  names->resize(2);
  (*names)[0] = "d0/d/h";
  (*names)[1] = "d0/d/c";
}

template <typename Dtype>
void ThreeDGridLSTMLayer<Dtype>::SetInputs(NetParameter* net_param, 
    const vector<Blob<Dtype>*>& bottom) const {
  BlobShape input_shape;
  // add input
  const int num_bottom = bottom.size(); 
  CHECK_LE(num_bottom, 2) << "num input must be less than equal to 2";
  if (num_bottom == 1) {
    input_shape.Clear();
    for (int i = 0; i < bottom[0]->num_axes(); ++i) {
      input_shape.add_dim(bottom[0]->shape(i));
    }
    net_param->add_input("x");
    net_param->add_input_shape()->CopyFrom(input_shape);
   
    LayerParameter split_layer_param;
    split_layer_param.set_type("Split");

    LayerParameter* x_copy_layer = net_param->add_layer();
    x_copy_layer->CopyFrom(split_layer_param);
    x_copy_layer->add_bottom("x");
    x_copy_layer->add_top("d0/d/h"); 
    x_copy_layer->add_top("d0/d/c");
  }
  else { // num_bottom == 2
    input_shape.Clear();
    for (int i = 0; i < bottom[0]->num_axes(); ++i) {
      input_shape.add_dim(bottom[0]->shape(i));
    }
    net_param->add_input("d0/d/h");
    net_param->add_input_shape()->CopyFrom(input_shape);

    //input_shape.Clear();
    //for (int i = 0; i < bottom[1]->num_axes(); ++i) {
    //  input_shape.add_dim(bottom[1]->shape(i));
    //}
    net_param->add_input("d0/d/c");
    net_param->add_input_shape()->CopyFrom(input_shape); // h and c should have the same input shape
  }
}

template <typename Dtype>
void ThreeDGridLSTMLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const {
  const int num_output = this->layer_param_.grid_param().num_output();
  CHECK_GT(num_output, 0) << "num_output must be positive";
  const FillerParameter& weight_filler =
      this->layer_param_.grid_param().weight_filler();
  const FillerParameter& bias_filler =
      this->layer_param_.grid_param().bias_filler();

  //bool is_peephole = this->layer_param_.grid_param().peephole(); 
  bool is_priority = this->layer_param_.grid_param().priority(); 

  // Add generic LayerParameter's (without bottoms/tops) of layer types we'll
  // use to save redundant code.
  LayerParameter conv_layer_param; 
  conv_layer_param.set_type("Convolution"); 
  conv_layer_param.mutable_convolution_param()->set_num_output(num_output * 3); // for memory cell (if peephole is true)
  conv_layer_param.mutable_convolution_param()->set_kernel_size(1); 
  conv_layer_param.mutable_convolution_param()->set_stride(1); 
  conv_layer_param.mutable_convolution_param()->set_bias_term(false);
  conv_layer_param.mutable_convolution_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);
  
  LayerParameter biased_conv_layer_param(conv_layer_param);
  biased_conv_layer_param.mutable_convolution_param()->set_num_output(num_output * 4 * 3); // for hidden state
  biased_conv_layer_param.mutable_convolution_param()->set_bias_term(true);
  biased_conv_layer_param.mutable_convolution_param()->
      mutable_bias_filler()->CopyFrom(bias_filler);

  LayerParameter hidden_layer_param;
  hidden_layer_param.set_type("InnerProduct");
  //hidden_layer_param.mutable_inner_product_param()->set_num_output(num_output * 4);
  hidden_layer_param.mutable_inner_product_param()->set_bias_term(false);
  //hidden_layer_param.mutable_inner_product_param()->set_axis(2);
  hidden_layer_param.mutable_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);

  //LayerParameter biased_hidden_layer_param(hidden_layer_param);
  //biased_hidden_layer_param.mutable_inner_product_param()->set_bias_term(true);
  //biased_hidden_layer_param.mutable_inner_product_param()->
  //    mutable_bias_filler()->CopyFrom(bias_filler);

  LayerParameter sum_layer_param;
  sum_layer_param.set_type("Eltwise");
  sum_layer_param.mutable_eltwise_param()->set_operation(
      EltwiseParameter_EltwiseOp_SUM);

  //LayerParameter scalar_layer_param;
  //scalar_layer_param.set_type("Scalar");
  //scalar_layer_param.mutable_scalar_param()->set_axis(0);

  LayerParameter slice_layer_param;
  slice_layer_param.set_type("Slice");
  //slice_layer_param.mutable_slice_param()->set_axis(0);

  LayerParameter split_layer_param;
  split_layer_param.set_type("Split");

  LayerParameter concat_layer_param;
  concat_layer_param.set_type("Concat");
  
  LayerParameter flatten_layer_param; 
  flatten_layer_param.set_type("Flatten"); 

  LayerParameter reshape_layer_param; 
  reshape_layer_param.set_type("Reshape");
  reshape_layer_param.mutable_reshape_param()->mutable_shape()->add_dim(0);
  reshape_layer_param.mutable_reshape_param()->mutable_shape()->add_dim(0);
  reshape_layer_param.mutable_reshape_param()->mutable_shape()->add_dim(1);
  reshape_layer_param.mutable_reshape_param()->mutable_shape()->add_dim(1);

  // Add c0s and h0s in height and width dimension. 
  vector<BlobShape> input_shapes;
  RecurrentInputShapes(&input_shapes);
  CHECK_EQ((this->height_ + this->width_) * 2, input_shapes.size()) << "In current implementation, RecurrentInputShapes should provide the shape of recurrent input c0 and h0, having zeros, in height and width dimension. ";
  vector<string> input_names;
  RecurrentInputBlobNames(&input_names);
  CHECK_EQ((this->height_ + this->width_) * 2, input_names.size());
  for (int i = 0; i < input_names.size(); ++i) {
    net_param->add_input(input_names[i]);
    net_param->add_input_shape()->CopyFrom(input_shapes[0]);
  }

  // Add layer to transform input hidden h to the transformed dimenstion.
  //     d0/d/h_transformed = W_h * d0/d/h + b_h
  {
    LayerParameter* h_transform_layer = net_param->add_layer();
    h_transform_layer->CopyFrom(biased_conv_layer_param);
    h_transform_layer->set_name("d0/d/h_transformed");
    h_transform_layer->add_param()->set_name("W_h/d");
    h_transform_layer->add_param()->set_name("b_h/d");
    h_transform_layer->add_bottom("d0/d/h");
    h_transform_layer->add_top("d0/d/h_transformed");
  }

  //// Add layer to transform input hidden c to the transformed dimension.
  ////     d0/d/h_transformed = W_h * d0/d/h + b_h
  //if (is_peephole) { // if peephole is true
  //  LayerParameter* c_transform_layer = net_param->add_layer();
  //  c_transform_layer->CopyFrom(conv_layer_param);
  //  c_transform_layer->set_name("d0/d/c_transform");
  //  c_transform_layer->add_param()->set_name("W_c/d");
  //  c_transform_layer->add_param()->set_name("b_c/d");
  //  c_transform_layer->add_bottom("d0/d/c");
  //  c_transform_layer->add_top("d0/d/c_transformed");
  //}

  // Slicing h_transformed along height dimension
  {
  LayerParameter* h_trans_slice_along_h_layer = net_param->add_layer();
  h_trans_slice_along_h_layer->CopyFrom(slice_layer_param);
  h_trans_slice_along_h_layer->mutable_slice_param()->set_axis(2);
  h_trans_slice_along_h_layer->set_name("slice/d0/d/h_transformed");
  h_trans_slice_along_h_layer->add_bottom("d0/d/h_transformed");
  for (int i = 0; i < this->height_; ++i) {
    h_trans_slice_along_h_layer->add_top("h"+this->int_to_str(i+1)+"_"+"d0/d/h_transformed");
  }
  }

  // (Further) slicing h_transformed along width dimension
  for (int i = 0; i < this->height_; ++i) {
    LayerParameter* h_trans_slice_along_w_layer = net_param->add_layer();
    h_trans_slice_along_w_layer->CopyFrom(slice_layer_param);
    h_trans_slice_along_w_layer->mutable_slice_param()->set_axis(3);
    h_trans_slice_along_w_layer->set_name("slice/h"+this->int_to_str(i+1)+"_"+"d0/d/h_transformed");
    h_trans_slice_along_w_layer->add_bottom("h"+this->int_to_str(i+1)+"_"+"d0/d/h_transformed");
    for (int j = 0; j < this->width_; ++j) {
      h_trans_slice_along_w_layer->add_top("h"+this->int_to_str(i+1)+"_"+
                                                 "w"+this->int_to_str(j+1)+"_"+
                                                 "d0/d/h_transformed");
    }
  }

  // (Further) slicing h_transformed along feature dimension
  for (int i = 0; i < this->height_; ++i) {
    for (int j = 0; j < this->width_; ++j) {
      LayerParameter* h_trans_slice_along_d_layer = net_param->add_layer();
      h_trans_slice_along_d_layer->CopyFrom(slice_layer_param);
      h_trans_slice_along_d_layer->mutable_slice_param()->set_axis(1);
      h_trans_slice_along_d_layer->mutable_slice_param()->add_slice_point(num_output * 4 * 2);
      h_trans_slice_along_d_layer->set_name("slice/h"+this->int_to_str(i+1)+"_"+
                                                  "w"+this->int_to_str(j+1)+"_"+
                                                  "d0/d/h_transformed");
      h_trans_slice_along_d_layer->mutable_inner_product_param()->set_axis(1);
      h_trans_slice_along_d_layer->add_bottom("h"+this->int_to_str(i+1)+"_"+
                                                    "w"+this->int_to_str(j+1)+"_"+
                                                    "d0/d/h_transformed");
      h_trans_slice_along_d_layer->add_top("h"+this->int_to_str(i+1)+"_"+
                                                 "w"+this->int_to_str(j+1)+"_"+
                                                 "d0/d/h_transformed/hw/preshape");
      h_trans_slice_along_d_layer->add_top("h"+this->int_to_str(i+1)+"_"+
                                                 "w"+this->int_to_str(j+1)+"_"+
                                                 "d0/d/h_transformed/d/preshape");
      {
      LayerParameter* flatten_layer = net_param->add_layer(); 
      flatten_layer->CopyFrom(flatten_layer_param);
      flatten_layer->set_name("h"+this->int_to_str(i+1)+"_"+
                              "w"+this->int_to_str(j+1)+"_"+
                              "d0/d/h_transformed/hw"); 
      flatten_layer->add_bottom("h"+this->int_to_str(i+1)+"_"+
                               "w"+this->int_to_str(j+1)+"_"+
                               "d0/d/h_transformed/hw/preshape");
      flatten_layer->add_top("h"+this->int_to_str(i+1)+"_"+
                             "w"+this->int_to_str(j+1)+"_"+
                             "d0/d/h_transformed/hw");
      }
      {
      LayerParameter* flatten_layer = net_param->add_layer();
      flatten_layer->CopyFrom(flatten_layer_param);
      flatten_layer->set_name("h"+this->int_to_str(i+1)+"_"+
                              "w"+this->int_to_str(j+1)+"_"+
                              "d0/d/h_transformed/hw");
      flatten_layer->add_bottom("h"+this->int_to_str(i+1)+"_"+
                                "w"+this->int_to_str(j+1)+"_"+
                                "d0/d/h_transformed/d/preshape");
      flatten_layer->add_top("h"+this->int_to_str(i+1)+"_"+
                             "w"+this->int_to_str(j+1)+"_"+
                             "d0/d/h_transformed/d");
      }
    }
  }

  // Slicing c along height dimention
  {
  LayerParameter* c_slice_along_h_layer = net_param->add_layer();
  c_slice_along_h_layer->CopyFrom(slice_layer_param);
  c_slice_along_h_layer->mutable_slice_param()->set_axis(2);
  c_slice_along_h_layer->set_name("slice/d0/d/c");
  c_slice_along_h_layer->add_bottom("d0/d/c");
  for (int i = 0; i < this->height_; ++i) {
    c_slice_along_h_layer->add_top("h"+this->int_to_str(i+1)+"_"+"d0/d/c");
  }
  }

  // (Further) slicing c along width dimension
  for (int i = 0; i < this->height_; ++i) {
    LayerParameter* c_slice_along_w_layer = net_param->add_layer();
    c_slice_along_w_layer->CopyFrom(slice_layer_param);
    c_slice_along_w_layer->mutable_slice_param()->set_axis(3);
    c_slice_along_w_layer->set_name("slice/h"+this->int_to_str(i+1)+"_"+"d0/d/c");
    c_slice_along_w_layer->add_bottom("h"+this->int_to_str(i+1)+"_"+"d0/d/c");
    for (int j = 0; j < this->width_; ++j) {
      c_slice_along_w_layer->add_top("h"+this->int_to_str(i+1)+"_"+
                                     "w"+this->int_to_str(j+1)+"_"+"d0/d/c/preshape");

      {
      LayerParameter* flatten_layer = net_param->add_layer();
      flatten_layer->CopyFrom(flatten_layer_param);
      flatten_layer->set_name("h"+this->int_to_str(i+1)+"_"+
                              "w"+this->int_to_str(j+1)+"_"+"d0/d/c");
      flatten_layer->add_bottom("h"+this->int_to_str(i+1)+"_"+
                                "w"+this->int_to_str(j+1)+"_"+"d0/d/c/preshape");
      flatten_layer->add_top("h"+this->int_to_str(i+1)+"_"+
                             "w"+this->int_to_str(j+1)+"_"+"d0/d/c");
      }
    }
  }

  //if (is_peephole) {
  //  // Slicing c_transformed along height dimention
  //  LayerParameter* c_trans_slice_along_h_layer = net_param->add_layer();
  //  c_trans_slice_along_h_layer->CopyFrom(slice_layer_param);
  //  c_trans_slice_along_h_layer->add_bottom("d0/d/c_transformed");
  //  for (int i = 0; i < this->height_; ++i) {
  //    c_trans_slice_along_h_layer->set_name("h"+this->int_to_str(i+1)+"_"+"d0/d/c_transformed");
  //  }
  //
  //  // (Further) slicing c_transformed along width dimension
  //  for (int i = 0; i < this->height_; ++i) {
  //    LayerParameter* c_trans_slice_along_w_layer = net_param->add_layer();
  //    c_trans_slice_along_w_layer->CopyFrom(slice_layer_param);
  //    c_trans_slice_along_w_layer->add_bottom("h"+this->int_to_str(i+1)+"_"+"d0/d/c_transformed");
  //    for (int j = 0; j < this->width_; ++j) {
  //      c_trans_slice_along_w_layer->set_name("h"+this->int_to_str(i+1)+"_"+
  //                                                  "w"+this->int_to_str(j+1)+"_"+"d0/d/c_transformed");
  //    }
  //  }
  //}

  // Propagate grid architecture
  for (int w = 1; w <= this->width_; ++w) {
    /**
     * Add grid architecture along height dimension 
     */
    for (int h = 1; h <= this->height_; ++h) {
      // Add inner product layer <h-1,w,d>/h/h_transformed/hw
      {
        LayerParameter* h_transform_layer = net_param->add_layer();
        h_transform_layer->CopyFrom(hidden_layer_param);
        h_transform_layer->mutable_inner_product_param()->set_num_output(num_output * 4 * 2);
        h_transform_layer->set_name("h"+this->int_to_str(h-1)+"_"+
                                    "w"+this->int_to_str(w)+"_"+
                                    "d1"+"/h/h_transformed/hw");
        h_transform_layer->add_param()->set_name("W_h/h/hw");
        h_transform_layer->add_bottom("h"+this->int_to_str(h-1)+"_"+
                                    "w"+this->int_to_str(w)+"_"+
                                    "d1"+"/h/h");
        h_transform_layer->add_top("h"+this->int_to_str(h-1)+"_"+
                                   "w"+this->int_to_str(w)+"_"+
                                   "d1"+"/h/h_transformed/hw");
      }
      
      // Add inner product layer <h-1,w,d>/h/h_transformed/d
      if (!is_priority || (h-1) != 0) {
        LayerParameter* h_transform_layer = net_param->add_layer();
        h_transform_layer->CopyFrom(hidden_layer_param);
        h_transform_layer->mutable_inner_product_param()->set_num_output(num_output * 4);
        h_transform_layer->set_name("h"+this->int_to_str(h-1)+"_"+
                                    "w"+this->int_to_str(w)+"_"+
                                    "d1"+"/h/h_transformed/d");
        h_transform_layer->add_param()->set_name("W_h/h/d");
        h_transform_layer->add_bottom("h"+this->int_to_str(h-1)+"_"+
                                    "w"+this->int_to_str(w)+"_"+
                                    "d1"+"/h/h");
        h_transform_layer->add_top("h"+this->int_to_str(h-1)+"_"+
                                   "w"+this->int_to_str(w)+"_"+
                                   "d1"+"/h/h_transformed/d");
      }

      // Add inner product layer <h,w-1,d>/w/h_transformed/hw
      {
        LayerParameter* h_transform_layer = net_param->add_layer();
        h_transform_layer->CopyFrom(hidden_layer_param);
        h_transform_layer->mutable_inner_product_param()->set_num_output(num_output * 4 * 2);
        h_transform_layer->set_name("h"+this->int_to_str(h)+"_"+
                                    "w"+this->int_to_str(w-1)+"_"+
                                    "d1"+"/w/h_transformed/hw");
        h_transform_layer->add_param()->set_name("W_h/w/hw");
        h_transform_layer->add_bottom("h"+this->int_to_str(h)+"_"+
                                    "w"+this->int_to_str(w-1)+"_"+
                                    "d1"+"/w/h");
        h_transform_layer->add_top("h"+this->int_to_str(h)+"_"+
                                   "w"+this->int_to_str(w-1)+"_"+
                                   "d1"+"/w/h_transformed/hw");
      }

      // Add inner product layer <h,w-1,d>/w/h_transformed/d
      if (!is_priority || (w-1) != 0) {
        LayerParameter* h_transform_layer = net_param->add_layer();
        h_transform_layer->CopyFrom(hidden_layer_param);
        h_transform_layer->mutable_inner_product_param()->set_num_output(num_output * 4);
        h_transform_layer->set_name("h"+this->int_to_str(h)+"_"+
                                    "w"+this->int_to_str(w-1)+"_"+
                                    "d1"+"/w/h_transformed/d");
        h_transform_layer->add_param()->set_name("W_h/w/d");
        h_transform_layer->add_bottom("h"+this->int_to_str(h)+"_"+
                                    "w"+this->int_to_str(w-1)+"_"+
                                    "d1"+"/w/h");
        h_transform_layer->add_top("h"+this->int_to_str(h)+"_"+
                                   "w"+this->int_to_str(w-1)+"_"+
                                   "d1"+"/w/h_transformed/d");
      }

      // Add elementwise operation layer <h,w,d>/hw/gate_input
      {
        LayerParameter* input_sum_layer = net_param->add_layer();
        input_sum_layer->CopyFrom(sum_layer_param);
        input_sum_layer->set_name("h"+this->int_to_str(h)+"_"+
                                  "w"+this->int_to_str(w)+"_"+
                                  "d1"+"/hw/gate_input");
        input_sum_layer->add_bottom("h"+this->int_to_str(h-1)+"_"+
                                    "w"+this->int_to_str(w)+"_"+
                                    "d1"+"/h/h_transformed/hw");
        input_sum_layer->add_bottom("h"+this->int_to_str(h)+"_"+
                                    "w"+this->int_to_str(w-1)+"_"+
                                    "d1"+"/w/h_transformed/hw");
        input_sum_layer->add_bottom("h"+this->int_to_str(h)+"_"+
                                    "w"+this->int_to_str(w)+"_"+
                                    "d0"+"/d/h_transformed/hw");
        input_sum_layer->add_top("h"+this->int_to_str(h)+"_"+
                                 "w"+this->int_to_str(w)+"_"+
                                 "d1"+"/hw/gate_input");
      }

      // Add slice layer
      {
      LayerParameter* gate_input_slice_layer = net_param->add_layer();
      gate_input_slice_layer->CopyFrom(slice_layer_param);
      gate_input_slice_layer->set_name("slice/h"+this->int_to_str(h)+"_"+
                                       "w"+this->int_to_str(w)+"_"+
                                       "d1"+"/hw/gate_input");
      gate_input_slice_layer->add_bottom("h"+this->int_to_str(h)+"_"+
                                         "w"+this->int_to_str(w)+"_"+
                                         "d1"+"/hw/gate_input");
      gate_input_slice_layer->add_top("h"+this->int_to_str(h)+"_"+
                                      "w"+this->int_to_str(w)+"_"+
                                      "d1"+"/h/gate_input");
      gate_input_slice_layer->add_top("h"+this->int_to_str(h)+"_"+
                                      "w"+this->int_to_str(w)+"_"+
                                      "d1"+"/w/gate_input");
      gate_input_slice_layer->mutable_slice_param()->set_axis(1);
      gate_input_slice_layer->mutable_slice_param()->add_slice_point(num_output * 4);
      }

      // Add LSTMUnit layer (along height dimension)
      {
      LayerParameter* lstm_unit_param = net_param->add_layer();
      lstm_unit_param->set_type("LSTMUnit");
      lstm_unit_param->add_bottom("h"+this->int_to_str(h-1)+"_"+
                                  "w"+this->int_to_str(w)+"_"+
                                  "d1"+"/h/c");
      lstm_unit_param->add_bottom("h"+this->int_to_str(h)+"_"+
                                 "w"+this->int_to_str(w)+"_"+
                                 "d1"+"/h/gate_input");
      lstm_unit_param->add_top("h"+this->int_to_str(h)+"_"+
                               "w"+this->int_to_str(w)+"_"+
                               "d1"+"/h/c");
      lstm_unit_param->add_top("h"+this->int_to_str(h)+"_"+
                               "w"+this->int_to_str(w)+"_"+
                               "d1"+"/h/h");
      lstm_unit_param->set_name("h"+this->int_to_str(h)+"_"+
                                "w"+this->int_to_str(w)+"_"+
                                "d1"+"/h");
      }
      
      if (is_priority && h == this->height_) {
      // Add inner product layer <h,w>/h/h_transformed/d
      LayerParameter* h_transform_layer = net_param->add_layer();
      h_transform_layer->CopyFrom(hidden_layer_param);
      h_transform_layer->mutable_inner_product_param()->set_num_output(num_output * 4);
      h_transform_layer->set_name("h"+this->int_to_str(h)+"_"+
                                  "w"+this->int_to_str(w)+"_"+
                                  "d1"+"/h/h_transformed/d");
      h_transform_layer->add_param()->set_name("W_h/h/d");
      h_transform_layer->add_bottom("h"+this->int_to_str(h)+"_"+
                                    "w"+this->int_to_str(w)+"_"+
                                    "d1"+"/h/h");
      h_transform_layer->add_top("h"+this->int_to_str(h)+"_"+
                                 "w"+this->int_to_str(w)+"_"+
                                 "d1"+"/h/h_transformed/d");
      }
    }
   
    /**
     * Add grid architecture along width dimension
     */ 
    for (int h = 1; h <= this->height_; ++h) {
      // Add LSTMUnit layer (along width dimension)
      {
      LayerParameter* lstm_unit_param = net_param->add_layer();
      lstm_unit_param->set_type("LSTMUnit");
      lstm_unit_param->add_bottom("h"+this->int_to_str(h)+"_"+
                                  "w"+this->int_to_str(w-1)+"_"+
                                  "d1"+"/w/c");
      lstm_unit_param->add_bottom("h"+this->int_to_str(h)+"_"+
                                 "w"+this->int_to_str(w)+"_"+
                                 "d1"+"/w/gate_input");
      lstm_unit_param->add_top("h"+this->int_to_str(h)+"_"+
                               "w"+this->int_to_str(w)+"_"+
                               "d1"+"/w/c");
      lstm_unit_param->add_top("h"+this->int_to_str(h)+"_"+
                               "w"+this->int_to_str(w)+"_"+
                               "d1"+"/w/h");
      lstm_unit_param->set_name("h"+this->int_to_str(h)+"_"+
                                "w"+this->int_to_str(w)+"_"+
                                "d1"+"/w");
      }

      // Add inner product layer <h,w>/w/h_transformed/d
      if (is_priority && w == this->width_) {
        LayerParameter* h_transform_layer = net_param->add_layer();
        h_transform_layer->CopyFrom(hidden_layer_param);
        h_transform_layer->mutable_inner_product_param()->set_num_output(num_output * 4);
        h_transform_layer->set_name("h"+this->int_to_str(h)+"_"+
                                    "w"+this->int_to_str(w)+"_"+
                                    "d1"+"/w/h_transformed/d");
        h_transform_layer->add_param()->set_name("W_h/w/d");
        h_transform_layer->add_bottom("h"+this->int_to_str(h)+"_"+
                                      "w"+this->int_to_str(w)+"_"+
                                      "d1"+"/w/h");
        h_transform_layer->add_top("h"+this->int_to_str(h)+"_"+
                                   "w"+this->int_to_str(w)+"_"+
                                   "d1"+"/w/h_transformed/d");
      }
    }
  }
  
  /**
   * Add grid architecture along depth dimension
   */
  for (int w = 1; w <= this->width_; ++w) {
    for (int h = 1; h <= this->height_; ++h) {
      // Add elementwise operation layer <h,w,d>/d/gate_input
      {
      LayerParameter* input_sum_layer = net_param->add_layer();
      input_sum_layer->CopyFrom(sum_layer_param);
      input_sum_layer->set_name("h"+this->int_to_str(h)+"_"+
                                "w"+this->int_to_str(w)+"_"+
                                "d1"+"/d/gate_input");
      if (is_priority) {
        input_sum_layer->add_bottom("h"+this->int_to_str(h)+"_"+
                                    "w"+this->int_to_str(w)+"_"+
                                    "d1"+"/h/h_transformed/d");
        input_sum_layer->add_bottom("h"+this->int_to_str(h)+"_"+
                                    "w"+this->int_to_str(w)+"_"+
                                    "d1"+"/w/h_transformed/d");
      } 
      else {
        input_sum_layer->add_bottom("h"+this->int_to_str(h-1)+"_"+
                                    "w"+this->int_to_str(w)+"_"+
                                    "d1"+"/h/h_transformed/d");
        input_sum_layer->add_bottom("h"+this->int_to_str(h)+"_"+
                                    "w"+this->int_to_str(w-1)+"_"+
                                    "d1"+"/w/h_transformed/d");
      }
      input_sum_layer->add_bottom("h"+this->int_to_str(h)+"_"+
                                  "w"+this->int_to_str(w)+"_"+
                                  "d0"+"/d/h_transformed/d");
      input_sum_layer->add_top("h"+this->int_to_str(h)+"_"+
                               "w"+this->int_to_str(w)+"_"+
                               "d1"+"/d/gate_input");
      }

      // Add LSTMUnit layer (along depth dimension) 
      {
      LayerParameter* lstm_unit_param = net_param->add_layer();
      lstm_unit_param->set_type("LSTMUnit");
      lstm_unit_param->add_bottom("h"+this->int_to_str(h)+"_"+
                                  "w"+this->int_to_str(w)+"_"+
                                  "d0"+"/d/c");
      lstm_unit_param->add_bottom("h"+this->int_to_str(h)+"_"+
                                 "w"+this->int_to_str(w)+"_"+
                                 "d1"+"/d/gate_input");
      lstm_unit_param->add_top("h"+this->int_to_str(h)+"_"+
                               "w"+this->int_to_str(w)+"_"+
                               "d1"+"/d/c");
      lstm_unit_param->add_top("h"+this->int_to_str(h)+"_"+
                               "w"+this->int_to_str(w)+"_"+
                               "d1"+"/d/h");
      lstm_unit_param->set_name("h"+this->int_to_str(h)+"_"+
                                "w"+this->int_to_str(w)+"_"+
                                "d1"+"/d");
      }
    }
  }
 
  // Reshape <h,w,d>/d/h and <h,w,d>/d/c of all lstm ouptuts, allowing them to be concatenated
  for (int h = 1; h <= this->height_; ++h) {
    for (int w = 1; w <= this->width_; ++w) {
      {
      LayerParameter* reshape_layer = net_param->add_layer(); 
      reshape_layer->CopyFrom(reshape_layer_param); 
      reshape_layer->set_name("h"+this->int_to_str(h)+"_"+
                              "w"+this->int_to_str(w)+"_"+
                              "d1"+"/d/h/reshape");  
      reshape_layer->add_bottom("h"+this->int_to_str(h)+"_"+
                                "w"+this->int_to_str(w)+"_"+
                                "d1"+"/d/h");
      reshape_layer->add_top("h"+this->int_to_str(h)+"_"+
                             "w"+this->int_to_str(w)+"_"+
                             "d1"+"/d/h/reshape");
      }
      {
      LayerParameter* reshape_layer = net_param->add_layer(); 
      reshape_layer->CopyFrom(reshape_layer_param); 
      reshape_layer->set_name("h"+this->int_to_str(h)+"_"+
                              "w"+this->int_to_str(w)+"_"+
                              "d1"+"/d/c/reshape");  
      reshape_layer->add_bottom("h"+this->int_to_str(h)+"_"+
                                "w"+this->int_to_str(w)+"_"+
                                "d1"+"/d/c");
      reshape_layer->add_top("h"+this->int_to_str(h)+"_"+
                             "w"+this->int_to_str(w)+"_"+
                             "d1"+"/d/c/reshape");
      }
    }
  }

  // Concatenate <h,w,d>/d/h and <h,w,d>/d/c of all lstm outputs
  // hidden
  // Concatenating <h,w,d>/d/h along width dimension
  for (int h = 1; h <= this->height_; ++h) {
    LayerParameter* concat_layer = net_param->add_layer(); 
    concat_layer->CopyFrom(concat_layer_param);
    concat_layer->set_name("h"+this->int_to_str(h)+"_"+
                           "d1"+"/d/h");
    concat_layer->mutable_concat_param()->set_axis(3);
    concat_layer->add_top("h"+this->int_to_str(h)+"_"+
                          "d1"+"/d/h");
    for (int w = 1; w <= this->width_; ++w) {
      concat_layer->add_bottom("h"+this->int_to_str(h)+"_"+
                               "w"+this->int_to_str(w)+"_"+
                               "d1"+"/d/h/reshape");
    }
  }

  // Concatenating <h,d>/d/h along height dimension
  {
  LayerParameter* concat_layer = net_param->add_layer();
  concat_layer->CopyFrom(concat_layer_param);
  concat_layer->set_name("d1/d/h");
  concat_layer->mutable_concat_param()->set_axis(2);
  concat_layer->add_top("d1/d/h");
  for (int h = 1; h <= this->height_; ++h) {
      concat_layer->add_bottom("h"+this->int_to_str(h)+"_"+
                               "d1"+"/d/h");
  }
  }

  // memory cell 
  // Concatenating <h,w,d>/d/c along width dimension
  for (int h = 1; h <= this->height_; ++h) {
    LayerParameter* concat_layer = net_param->add_layer();
    concat_layer->CopyFrom(concat_layer_param);
    concat_layer->set_name("h"+this->int_to_str(h)+"_"+
                           "d1"+"/d/c");
    concat_layer->mutable_concat_param()->set_axis(3);
    concat_layer->add_top("h"+this->int_to_str(h)+"_"+
                          "d1"+"/d/c");
    for (int w = 1; w <= this->width_; ++w) {
      concat_layer->add_bottom("h"+this->int_to_str(h)+"_"+
                               "w"+this->int_to_str(w)+"_"+
                               "d1"+"/d/c/reshape");
    }
  }

  // Concatenating <h,d>/d/h along height dimension
  {
  LayerParameter* concat_layer = net_param->add_layer();
  concat_layer->CopyFrom(concat_layer_param);
  concat_layer->set_name("d1/d/c");
  concat_layer->mutable_concat_param()->set_axis(2);
  concat_layer->add_top("d1/d/c");
  for (int h = 1; h <= this->height_; ++h) {
      concat_layer->add_bottom("h"+this->int_to_str(h)+"_"+
                               "d1"+"/d/c");
  }
  }
}

INSTANTIATE_CLASS(ThreeDGridLSTMLayer);
REGISTER_LAYER_CLASS(ThreeDGridLSTM);

}  // namespace caffe
