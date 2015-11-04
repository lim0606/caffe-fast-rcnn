#ifndef CAFFE_GRID_LAYERS_HPP_
#define CAFFE_GRID_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

//template <typename Dtype> class GridLayer;
template <typename Dtype> class ThreeDGridLayer;

/**
 * @brief An abstract class for implementing recurrent behavior inside of an
 *        unrolled network.  This Layer type cannot be instantiated -- instaed,
 *        you should use one of its implementations which defines the recurrent
 *        architecture, such as RNNLayer or LSTMLayer.
 */
template <typename Dtype>
class ThreeDGridLayer : public Layer<Dtype> {
 public:
  explicit ThreeDGridLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Reset();

  virtual inline const char* type() const { return "ThreeDGrid"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  //virtual inline int MaxBottomBlobs() const { return 3; }
  //virtual inline int ExactNumTopBlobs() const { return 1; }

  //virtual inline bool AllowForceBackward(const int bottom_index) const {
  //  // Can't propagate to sequence continuation indicators.
  //  return bottom_index != 1;
  //}

 protected:
  /**
   * @brief Fills net_param with the recurrent network arcthiecture.  Subclasses
   *        should define this -- see RNNLayer and LSTMLayer for examples.
   */
  virtual void SetInputs(NetParameter* net_param, const vector<Blob<Dtype>*>& bottom) const;

  /**
   * @brief Fills net_param with the recurrent network arcthiecture.  Subclasses
   *        should define this -- see RNNLayer and LSTMLayer for examples.
   */
  virtual void FillUnrolledNet(NetParameter* net_param) const = 0;

  /**
   * @brief Fills names with the names of the 0th timestep recurrent input
   *        Blob&s.  Subclasses should define this -- see RNNLayer and LSTMLayer
   *        for examples.
   */
  virtual void RecurrentInputBlobNames(vector<string>* names) const = 0;

  /**
   * @brief Fills shapes with the shapes of the recurrent input Blob&s.
   *        Subclasses should define this -- see RNNLayer and LSTMLayer
   *        for examples.
   */
  virtual void RecurrentInputShapes(vector<BlobShape>* shapes) const = 0;

  /**
   * @brief Fills names with the names of the Tth timestep recurrent output
   *        Blob&s.  Subclasses should define this -- see RNNLayer and LSTMLayer
   *        for examples.
   */
  //virtual void RecurrentOutputBlobNames(vector<string>* names) const = 0;

  /**
   * @brief added by jhlim
   */
  virtual void InputBlobNames(vector<string>* names, const int num_bottom) const = 0;

  /**
   * @brief Fills names with the names of the output blobs, concatenated across
   *        all timesteps.  Should return a name for each top Blob.
   *        Subclasses should define this -- see RNNLayer and LSTMLayer for
   *        examples.
   */
  virtual void OutputBlobNames(vector<string>* names) const = 0;

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// @brief A helper function, useful for stringifying timestep indices.
  virtual string int_to_str(const int t) const;

  /// @brief A Net to implement the ThreeDGrid functionality.
  shared_ptr<Net<Dtype> > unrolled_net_;

  vector<Blob<Dtype>* > recur_input_blobs_;
  //vector<Blob<Dtype>* > recur_output_blobs_;
  vector<Blob<Dtype>* > input_blobs_;
  vector<Blob<Dtype>* > output_blobs_;

  /**
   * @brief asdf
   */
  int num_;
  int channels_;
  int height_;
  int width_;

  class Num {
   public: 
    Num(const int size, const int inc = 1) {
      init(size, inc); 
    }
    inline void init(const int size, const int inc = 1) {
      CHECK_EQ(inc*inc, 1) << "inc should be either 1 or -1";
      inc_ = inc; 
      size_ = size; 
      if (inc == 1) {
        start_ = 1;
        end_ = size_;  
      }
      else { // inc_
        start_ = size_;
        end_ = 1;  
      } 
      val_ = start_; 
    }
    inline int next() { return val_+inc_; }
    inline int prev() { return val_-inc_; }
    inline void move_next() { val_=val_+inc_; }
    inline bool is_within_range() { 
        return (1 <= val_ && val_ <= size_); 
    }
    inline int val() { return val_; }
    inline int start() { return start_; }
    inline int end() { return end_; }
    inline int start_prev() { return start_-inc_; }
    
   protected: 
    int val_; 
    int inc_;
    int size_; 
    int start_;
    int end_;  
  };

  shared_ptr<Num> h_; 
  shared_ptr<Num> w_;
  int h_inc_; 
  int w_inc_;  
};


///**
// * @brief An abstract class for implementing recurrent behavior inside of an
// *        unrolled network.  This Layer type cannot be instantiated -- instaed,
// *        you should use one of its implementations which defines the recurrent
// *        architecture, such as RNNLayer or LSTMLayer.
// */
//template <typename Dtype>
//class ThreeDGridLayer : public GridLayer<Dtype> {
// public:
//  explicit ThreeDGridLayer(const LayerParameter& param)
//      : GridLayer<Dtype>(param) {}
//  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
//      const vector<Blob<Dtype>*>& top);
//  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
//      const vector<Blob<Dtype>*>& top);
//  virtual void Reset();
//
//  virtual inline const char* type() const { return "ThreeDGrid"; }
//  virtual inline int MinBottomBlobs() const { return 1; }
//  //virtual inline int MaxBottomBlobs() const { return 3; }
//  //virtual inline int ExactNumTopBlobs() const { return 1; }
//
// protected:
//  virtual void SetInputs(NetParameter* net_param, const vector<Blob<Dtype>*>& bottom) const;
//  virtual void FillUnrolledNet(NetParameter* net_param) const;
//  virtual void RecurrentInputBlobNames(vector<string>* names) const;
//  virtual void RecurrentInputShapes(vector<BlobShape>* shapes) const;
//  virtual void RecurrentOutputBlobNames(vector<string>* names) const;
//  virtual void InputBlobNames(vector<string>* names, const int num_bottom) const;
//  virtual void OutputBlobNames(vector<string>* names) const;
//
//  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//      const vector<Blob<Dtype>*>& top);
//  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//      const vector<Blob<Dtype>*>& top);
//  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
//      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
//
//  virtual string int_to_str(const int t) const;
//
//  /**
//   * @brief asdf
//   */ 
//  int num_; 
//  int channels_; 
//  int height_; 
//  int width_; 
//};

/**
 * @brief Processes sequential inputs using a "Long Short-Term Memory" (LSTM)
 *        [1] style recurrent neural network (RNN). Implemented as a network
 *        unrolled the LSTM computation in time.
 *
 *
 * The specific architecture used in this implementation is as described in
 * "Learning to Execute" [2], reproduced below:
 *     i_t := \sigmoid[ W_{hi} * h_{t-1} + W_{xi} * x_t + b_i ]
 *     f_t := \sigmoid[ W_{hf} * h_{t-1} + W_{xf} * x_t + b_f ]
 *     o_t := \sigmoid[ W_{ho} * h_{t-1} + W_{xo} * x_t + b_o ]
 *     g_t :=    \tanh[ W_{hg} * h_{t-1} + W_{xg} * x_t + b_g ]
 *     c_t := (f_t .* c_{t-1}) + (i_t .* g_t)
 *     h_t := o_t .* \tanh[c_t]
 * In the implementation, the i, f, o, and g computations are performed as a
 * single inner product.
 *
 * Notably, this implementation lacks the "diagonal" gates, as used in the
 * LSTM architectures described by Alex Graves [3] and others.
 *
 * [1] Hochreiter, Sepp, and Schmidhuber, Jürgen. "Long short-term memory."
 *     Neural Computation 9, no. 8 (1997): 1735-1780.
 *
 * [2] Zaremba, Wojciech, and Sutskever, Ilya. "Learning to execute."
 *     arXiv preprint arXiv:1410.4615 (2014).
 *
 * [3] Graves, Alex. "Generating sequences with recurrent neural networks."
 *     arXiv preprint arXiv:1308.0850 (2013).
 */
template <typename Dtype>
class ThreeDGridLSTMLayer : public ThreeDGridLayer<Dtype> {
 public:
  explicit ThreeDGridLSTMLayer(const LayerParameter& param)
      : ThreeDGridLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "ThreeDGridLSTM"; }

 protected:
  virtual void SetInputs(NetParameter* net_param, const vector<Blob<Dtype>*>& bottom) const;
  virtual void FillUnrolledNet(NetParameter* net_param) const;
  virtual void RecurrentInputBlobNames(vector<string>* names) const;
  virtual void RecurrentInputShapes(vector<BlobShape>* shapes) const;
  //virtual void RecurrentOutputBlobNames(vector<string>* names) const;
  virtual void InputBlobNames(vector<string>* names, const int num_bottom) const;
  virtual void OutputBlobNames(vector<string>* names) const;

};

/**
 * @brief A helper for LSTMLayer: computes a single timestep of the
 *        non-linearity of the LSTM, producing the updated cell and hidden
 *        states.
 */
template <typename Dtype>
class LSTMUnitLayer : public Layer<Dtype> {
 public:
  explicit LSTMUnitLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LSTMUnit"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  /**
   * @param bottom input Blob vector (length 3)
   *   -# @f$ (1 \times N \times D) @f$
   *      the previous timestep cell state @f$ c_{t-1} @f$
   *   -# @f$ (1 \times N \times 4D) @f$
   *      the "gate inputs" @f$ [i_t', f_t', o_t', g_t'] @f$
   *   -# @f$ (1 \times N) @f$
   *      the sequence continuation indicators  @f$ \delta_t @f$
   * @param top output Blob vector (length 2)
   *   -# @f$ (1 \times N \times D) @f$
   *      the updated cell state @f$ c_t @f$, computed as:
   *          i_t := \sigmoid[i_t']
   *          f_t := \sigmoid[f_t']
   *          o_t := \sigmoid[o_t']
   *          g_t := \tanh[g_t']
   *          c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
   *   -# @f$ (1 \times N \times D) @f$
   *      the updated hidden state @f$ h_t @f$, computed as:
   *          h_t := o_t .* \tanh[c_t]
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the LSTMUnit inputs.
   *
   * @param top output Blob vector (length 2), providing the error gradient with
   *        respect to the outputs
   *   -# @f$ (1 \times N \times D) @f$:
   *      containing error gradients @f$ \frac{\partial E}{\partial c_t} @f$
   *      with respect to the updated cell state @f$ c_t @f$
   *   -# @f$ (1 \times N \times D) @f$:
   *      containing error gradients @f$ \frac{\partial E}{\partial h_t} @f$
   *      with respect to the updated cell state @f$ h_t @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 3), into which the error gradients
   *        with respect to the LSTMUnit inputs @f$ c_{t-1} @f$ and the gate
   *        inputs are computed.  Computatation of the error gradients w.r.t.
   *        the sequence indicators is not implemented.
   *   -# @f$ (1 \times N \times D) @f$
   *      the error gradient w.r.t. the previous timestep cell state
   *      @f$ c_{t-1} @f$
   *   -# @f$ (1 \times N \times 4D) @f$
   *      the error gradient w.r.t. the "gate inputs"
   *      @f$ [
   *          \frac{\partial E}{\partial i_t}
   *          \frac{\partial E}{\partial f_t}
   *          \frac{\partial E}{\partial o_t}
   *          \frac{\partial E}{\partial g_t}
   *          ] @f$
   *   -# @f$ (1 \times 1 \times N) @f$
   *      the gradient w.r.t. the sequence continuation indicators
   *      @f$ \delta_t @f$ is currently not computed.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// @brief The hidden and output dimension.
  int hidden_dim_;
  Blob<Dtype> X_acts_;
};

///**
// * @brief Processes time-varying inputs using a simple recurrent neural network
// *        (RNN). Implemented as a network unrolling the RNN computation in time.
// *
// * Given time-varying inputs @f$ x_t @f$, computes hidden state @f$
// *     h_t := \tanh[ W_{hh} h_{t_1} + W_{xh} x_t + b_h ]
// * @f$, and outputs @f$
// *     o_t := \tanh[ W_{ho} h_t + b_o ]
// * @f$.
// */
//template <typename Dtype>
//class RNNLayer : public ThreeDGridLayer<Dtype> {
// public:
//  explicit RNNLayer(const LayerParameter& param)
//      : ThreeDGridLayer<Dtype>(param) {}
//
//  virtual inline const char* type() const { return "RNN"; }
//
// protected:
//  virtual void FillUnrolledNet(NetParameter* net_param) const;
//  virtual void RecurrentInputBlobNames(vector<string>* names) const;
//  virtual void RecurrentOutputBlobNames(vector<string>* names) const;
//  virtual void RecurrentInputShapes(vector<BlobShape>* shapes) const;
//  virtual void OutputBlobNames(vector<string>* names) const;
//};

}  // namespace caffe

#endif  // CAFFE_GRID_LAYERS_HPP_
