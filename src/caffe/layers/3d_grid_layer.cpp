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
string ThreeDGridLayer<Dtype>::int_to_str(const int t) const {
  ostringstream num;
  num << t;
  return num.str();
}

template <typename Dtype>
void ThreeDGridLayer<Dtype>::SetInputs(NetParameter* net_param,
    const vector<Blob<Dtype>*>& bottom) const {
  CHECK_EQ(bottom.size(), 1) << "Fail in initializing inputs of 3d grid layer with default SetInputs function. (" << bottom.size() << " number of bottoms are given)";
 
  net_param->add_input("x");
  BlobShape input_shape;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    input_shape.add_dim(bottom[0]->shape(i));
  }
  net_param->add_input_shape()->CopyFrom(input_shape);

  //input_shape.Clear();
  //for (int i = 0; i < bottom[1]->num_axes(); ++i) {
  //  input_shape.add_dim(bottom[1]->shape(i));
  //}
  //net_param->add_input("cont");
  //net_param->add_input_shape()->CopyFrom(input_shape);

  //if (static_input_) {
  //  input_shape.Clear();
  //  for (int i = 0; i < bottom[2]->num_axes(); ++i) {
  //    input_shape.add_dim(bottom[2]->shape(i));
  //  }
  //  net_param->add_input("x_static");
  //  net_param->add_input_shape()->CopyFrom(input_shape);
  //}

  LOG(INFO) << "Initializing input of 3d grid layer with default SetInputs function.";
}

template <typename Dtype>
void ThreeDGridLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_LE(bottom.size(), 2); 
  CHECK_EQ(bottom[0]->num_axes(), 4)
      << "bottom[0] must have at least 4 axes";
  num_ = bottom[0]->shape(0);
  channels_ = bottom[0]->shape(1);
  height_ = bottom[0]->shape(2); 
  width_ = bottom[0]->shape(3); 
  LOG(INFO) << "Initializing grid layer: assuming input contains "
            << num_ << " number of images."; 

  if (bottom.size() > 1) {
    CHECK_EQ(bottom[1]->num_axes(), 4)
        << "bottom[1] must have exactly 4 axes";
  }

  // Determind propagation direction
  ThreeDGridParameter_PropagateDirection prop_direction = this->layer_param_.grid_param().prop_direction();
  switch (prop_direction) {
  case ThreeDGridParameter_PropagateDirection_H_INC_W_INC:
    h_inc_ = 1; w_inc_ = 1;
    break;
  case ThreeDGridParameter_PropagateDirection_H_INC_W_DEC:
    h_inc_ = 1; w_inc_ = -1;
    break;
  case ThreeDGridParameter_PropagateDirection_H_DEC_W_INC:
    h_inc_ = -1; w_inc_ = 1;
    break;
  case ThreeDGridParameter_PropagateDirection_H_DEC_W_DEC:
    h_inc_ = -1; w_inc_ = -1;
    break;
  default:
    LOG(FATAL) << "Unknown propagate direction.";
  }
  h_.reset(new Num(height_, h_inc_));
  w_.reset(new Num(width_, w_inc_)); 

  // Create a NetParameter; setup the inputs that aren't unique to particular
  // recurrent architectures.
  NetParameter net_param;
  net_param.set_force_backward(true);

  // Call the child's SetInputs implementation to specify inputs for the unrolled
  // recurrent architecture. If there is not SetInputs of child's, call ThreeDGridLayer class's one.
  this->SetInputs(&net_param, bottom);
 
  // Call the child's FillUnrolledNet implementation to specify the unrolled
  // recurrent architecture.
  this->FillUnrolledNet(&net_param);

  // Prepend this layer's name to the names of each layer in the unrolled net.
  const string& layer_name = this->layer_param_.name();
  if (layer_name.size() > 0) {
    for (int i = 0; i < net_param.layer_size(); ++i) {
      LayerParameter* layer = net_param.mutable_layer(i);
      layer->set_name(layer_name + "_" + layer->name());
    }
  }

  // Create the unrolled net.
  this->unrolled_net_.reset(new Net<Dtype>(net_param));
  this->unrolled_net_->set_debug_info(this->layer_param_.grid_param().debug_info());

  //// Setup pointers to the inputs.
  //x_input_blob_ = CHECK_NOTNULL(this->unrolled_net_->blob_by_name("x").get());
  //cont_input_blob_ = CHECK_NOTNULL(this->unrolled_net_->blob_by_name("cont").get());
  //if (static_input_) {
  //  x_static_input_blob_ =
  //      CHECK_NOTNULL(this->unrolled_net_->blob_by_name("x_static").get());
  //}

  // Setup pointers to paired recurrent inputs/outputs.
  vector<string> recur_input_names;
  RecurrentInputBlobNames(&recur_input_names);
  //vector<string> recur_output_names;
  //RecurrentOutputBlobNames(&recur_output_names);
  const int num_recur_blobs = recur_input_names.size();
  //CHECK_EQ(num_recur_blobs, recur_output_names.size());
  this->recur_input_blobs_.resize(num_recur_blobs);
  //recur_output_blobs_.resize(num_recur_blobs);
  for (int i = 0; i < recur_input_names.size(); ++i) {
    this->recur_input_blobs_[i] =
        CHECK_NOTNULL(this->unrolled_net_->blob_by_name(recur_input_names[i]).get());
  //  recur_output_blobs_[i] =
  //      CHECK_NOTNULL(this->unrolled_net_->blob_by_name(recur_output_names[i]).get());
  }

  // Setup pointers to inputs.
  vector<string> input_names;
  InputBlobNames(&input_names, bottom.size());
  //CHECK_EQ(bottom.size(), input_names.size())
  //    << "InputBlobNames must provide an input blob name for each bottom.";
  this->input_blobs_.resize(input_names.size());
  for (int i = 0; i < input_names.size(); ++i) {
    this->input_blobs_[i] =
        CHECK_NOTNULL(this->unrolled_net_->blob_by_name(input_names[i]).get());
  }

  // Setup pointers to outputs.
  vector<string> output_names;
  OutputBlobNames(&output_names);
  CHECK_EQ(top.size(), output_names.size())
      << "OutputBlobNames must provide an output blob name for each top.";
  this->output_blobs_.resize(output_names.size());
  for (int i = 0; i < output_names.size(); ++i) {
    this->output_blobs_[i] =
        CHECK_NOTNULL(this->unrolled_net_->blob_by_name(output_names[i]).get());
  }

  // We should have one or two inputs (x or d0/d/c and d0/d/h), plus a number of recurrent inputs
  CHECK_EQ(bottom.size() + num_recur_blobs,
           this->unrolled_net_->input_blobs().size());

  // This layer's parameters are any parameters in the layers of the unrolled
  // net. We only want one copy of each parameter, so check that the parameter
  // is "owned" by the layer, rather than shared with another.
  this->blobs_.clear();
  for (int i = 0; i < this->unrolled_net_->params().size(); ++i) {
    if (this->unrolled_net_->param_owners()[i] == -1) {
      LOG(INFO) << "Adding parameter " << i << ": "
                << this->unrolled_net_->param_display_names()[i];
      this->blobs_.push_back(this->unrolled_net_->params()[i]);
    }
  }
  // Check that param_propagate_down is set for all of the parameters in the
  // unrolled net; set param_propagate_down to true in this layer.
  for (int i = 0; i < this->unrolled_net_->layers().size(); ++i) {
    for (int j = 0; j < this->unrolled_net_->layers()[i]->blobs().size(); ++j) {
      CHECK(this->unrolled_net_->layers()[i]->param_propagate_down(j))
          << "param_propagate_down not set for layer " << i << ", param " << j;
    }
  }
  this->param_propagate_down_.clear();
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void ThreeDGridLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    CHECK_EQ(bottom[i]->num_axes(), 4)
      << "bottom[" << i << "] must have exactly 4 axes " 
      << "-- (#batchsize, #channels, #height, #width)";
  }
  num_ = bottom[0]->shape(0); 
  channels_ = bottom[0]->shape(1); 
  height_ = bottom[0]->shape(2); 
  width_ = bottom[0]->shape(3); 

  CHECK_EQ(top.size(), this->output_blobs_.size());
  for (int i = 0; i < this->output_blobs_.size(); ++i) {
    CHECK_EQ(bottom[0]->shape(1), this->output_blobs_[i]->channels()); 
  }
  //x_input_blob_->ReshapeLike(*bottom[0]);
  //vector<int> cont_shape = bottom[1]->shape();
  //cont_input_blob_->Reshape(cont_shape);
  //if (static_input_) {
  //  x_static_input_blob_->ReshapeLike(*bottom[2]);
  //}
  for (int i = 0; i < bottom.size(); ++i) {
    this->input_blobs_[i]->ReshapeLike(*bottom[i]);
    this->input_blobs_[i]->ShareData(*bottom[i]);
    this->input_blobs_[i]->ShareDiff(*bottom[i]);
  }

  vector<BlobShape> recur_input_shapes;
  RecurrentInputShapes(&recur_input_shapes);
  this->blobs_[0]->Reshape(recur_input_shapes[0]);
  vector<string> recur_input_names;
  RecurrentInputBlobNames(&recur_input_names);
  CHECK_EQ(recur_input_names.size(), this->recur_input_blobs_.size());
  for (int i = 0; i < recur_input_names.size(); ++i) {
    this->recur_input_blobs_[i]->ReshapeLike(*(this->blobs_[0]));
    this->recur_input_blobs_[i]->ShareData(*(this->blobs_[0]));
    this->recur_input_blobs_[i]->ShareDiff(*(this->blobs_[0]));
  }
  this->unrolled_net_->Reshape();
  
  //x_input_blob_->ShareData(*bottom[0]);
  //x_input_blob_->ShareDiff(*bottom[0]);
  //cont_input_blob_->ShareData(*bottom[1]);
  //if (static_input_) {
  //  x_static_input_blob_->ShareData(*bottom[2]);
  //  x_static_input_blob_->ShareDiff(*bottom[2]);
  //}
  for (int i = 0; i < top.size(); ++i) {
    top[i]->ReshapeLike(*(this->output_blobs_[i]));
    top[i]->ShareData(*(this->output_blobs_[i]));
    top[i]->ShareDiff(*(this->output_blobs_[i]));
  }
}

//template <typename Dtype>
//void ThreeDGridLayer<Dtype>::Reset() {
//  // "Reset" the hidden state of the net by zeroing out all recurrent outputs.
//  for (int i = 0; i < recur_output_blobs_.size(); ++i) {
//    caffe_set(recur_output_blobs_[i]->count(), Dtype(0),
//              recur_output_blobs_[i]->mutable_cpu_data());
//  }
//}

template <typename Dtype>
void ThreeDGridLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //DCHECK_EQ(this->recur_input_blobs_.size(), recur_output_blobs_.size());
  //for (int i = 0; i < this->recur_input_blobs_.size(); ++i) {
  //  const int count = this->recur_input_blobs_[i]->count();
  //  DCHECK_EQ(count, recur_output_blobs_[i]->count());
  //  const Dtype* timestep_T_data = recur_output_blobs_[i]->cpu_data();
  //  Dtype* timestep_0_data = this->recur_input_blobs_[i]->mutable_cpu_data();
  //  caffe_copy(count, timestep_T_data, timestep_0_data);
  //}

  this->unrolled_net_->ForwardPrefilled();
}

template <typename Dtype>
void ThreeDGridLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //CHECK(!propagate_down[1]) << "Cannot backpropagate to sequence indicators.";

  // TODO: skip backpropagation to inputs and parameters inside the unrolled
  // net according to propagate_down[0] and propagate_down[2]. For now just
  // backprop to inputs and parameters unconditionally, as either the inputs or
  // the parameters do need backward (or Net would have set
  // layer_needs_backward_[i] == false for this layer).
  this->unrolled_net_->Backward();
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(ThreeDGridLayer, Forward);
#endif

INSTANTIATE_CLASS(ThreeDGridLayer);
//REGISTER_LAYER_CLASS(ThreeDGrid);

}  // namespace caffe
