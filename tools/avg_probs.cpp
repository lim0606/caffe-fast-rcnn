//Jaehyun Lim

#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <fstream> 
#include <sstream> 

#include "hdf5.h"
#include "leveldb/db.h"
#include "lmdb.h"

#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include <boost/shared_ptr.hpp>
#include <boost/pointer_cast.hpp>

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::BNLayer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;
//using caffe::LayerParameter_LayerType_BN; 
using caffe::caffe_set; 
using caffe::NetParameter; 
using boost::dynamic_pointer_cast;

// Define flags
DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
//DEFINE_string(solver, "",
//    "The solver definition protocol buffer text file.");
//DEFINE_string(train_model, "",
//    "The model definition protocol buffer text file..");
//DEFINE_string(test_model, "",
//    "The model definition protocol buffer text file..");
//DEFINE_string(snapshot, "",
//    "The snapshot solver state to resume training.");
//DEFINE_string(weights, "",
//    "The pretrained weights to initialize finetuning. "
//    "Cannot be set simultaneously with snapshot.");
//DEFINE_int32(train_iterations, 0,
//    "The number of iterations to run.");
//DEFINE_int32(numdata, 0,
//    "The total number of test data. (you should specify in this implementation)."); 
//DEFINE_int32(batchsize, 0,
//    "The batchsize. (you should specify in this implementation)."); 
//DEFINE_string(labellist, "",
//    "The text file having labels and their corresponding indices.");
DEFINE_string(outfile, "",
    "The text file including prediction probabilities.");
//DEFINE_string(target_blob, "prob",
//    "The name of blob you want to print out.");
//DEFINE_string(savefolder, "",
//    "The folder path that batch mean and batch variance should be stored.");
DEFINE_string(filelist, "",
    "The test file, each line of which indicates a file contains probs.");

std::string int_to_str(const int t) {
  std::ostringstream num;
  num << t;
  return num.str();
}

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Usage message.
  gflags::SetUsageMessage("\n"
      "usage: save_bn <args>\n\n");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 4) {
    //return GetBrewFunction(caffe::string(argv[1]))();
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/save_bn");
  }

  // Open file list
  std::ifstream filelist;
  filelist.open(FLAGS_filelist.c_str());
  if(!filelist) {
    LOG(FATAL) << "Please specify the label list file. For example, ndsb_labels.txt.\n";
    return 0;
  }
  std::vector< std::string > filenames;
  int num_files;  
  std::string filename;
  while(filelist >> filename) {
    //printf("filename: %s\n", filename.c_str());
    filenames.push_back(filename);
  }
  num_files = filenames.size();
  LOG(INFO) << "# of files : " << num_files;

  int numdata = 0; 
  int num_classes = 0; 
  {
  // Read proto for init numdata and num_classes  
  caffe::BlobProtoVector blob_proto_vec_tmp;
  blob_proto_vec_tmp.Clear();
  ReadProtoFromBinaryFile(filenames[0], &blob_proto_vec_tmp);

  const caffe::BlobProto& blob_proto_prob = blob_proto_vec_tmp.blobs(0); 
  const caffe::BlobProto& blob_proto_label = blob_proto_vec_tmp.blobs(1);

  Blob<float> input_blob_prob(1, 1, 1, 1); 
  Blob<float> input_blob_label(1, 1, 1, 1); 
 
  input_blob_prob.FromProto(blob_proto_prob, true);
  input_blob_label.FromProto(blob_proto_label, true);

  /*int*/ numdata = input_blob_prob.shape(0); 
  /*int*/ num_classes = input_blob_prob.shape(1);
  CHECK_EQ(numdata, input_blob_label.shape(0)); 
  }

  LOG(INFO) << "numdata: " << numdata;
  LOG(INFO) << "num_classes: " << num_classes;

  // init output blob
  Blob<float> output_blob_prob(numdata, num_classes, 1, 1); 
  Blob<float> output_blob_label(numdata, 1, 1, 1); 

  float* output_blob_prob_data = output_blob_prob.mutable_cpu_data();
  float* output_blob_label_data = output_blob_label.mutable_cpu_data();

  caffe_set(output_blob_prob.count(), float(1), output_blob_prob_data);

  //printf("# of iterations: %d\n", FLAGS_iterations);
  LOG(INFO) << "Start prediction";

  {
  caffe::BlobProtoVector blob_proto_vec_tmp;
  blob_proto_vec_tmp.Clear();
  ReadProtoFromBinaryFile(filenames[0], &blob_proto_vec_tmp);

  const caffe::BlobProto& blob_proto_prob = blob_proto_vec_tmp.blobs(0);
  const caffe::BlobProto& blob_proto_label = blob_proto_vec_tmp.blobs(1);

  Blob<float> input_blob_prob(1, 1, 1, 1);
  Blob<float> input_blob_label(1, 1, 1, 1);

  input_blob_prob.FromProto(blob_proto_prob, true);
  input_blob_label.FromProto(blob_proto_label, true);

  CHECK_EQ(numdata, input_blob_prob.shape(0));
  CHECK_EQ(num_classes, input_blob_prob.shape(1));

  const float* input_blob_prob_data = input_blob_prob.mutable_cpu_data();
  const float* input_blob_label_data = input_blob_label.mutable_cpu_data();

  for (int j = 0; j < numdata; ++j) {
    for (int k = 0; k < num_classes; ++k) {
      output_blob_prob_data[j*num_classes+k] = input_blob_prob_data[j*num_classes+k] / (static_cast<float>(num_files));
    }
    output_blob_label_data[j] = input_blob_label_data[j]; 
  }
  }

  for (int i = 1; i < num_files; ++i) {
    caffe::BlobProtoVector blob_proto_vec_tmp;
    blob_proto_vec_tmp.Clear();
    ReadProtoFromBinaryFile(filenames[i], &blob_proto_vec_tmp);
  
    const caffe::BlobProto& blob_proto_prob = blob_proto_vec_tmp.blobs(0);
    const caffe::BlobProto& blob_proto_label = blob_proto_vec_tmp.blobs(1);
  
    Blob<float> input_blob_prob(1, 1, 1, 1);
    Blob<float> input_blob_label(1, 1, 1, 1);
  
    input_blob_prob.FromProto(blob_proto_prob, true);
    input_blob_label.FromProto(blob_proto_label, true);
  
    CHECK_EQ(numdata, input_blob_prob.shape(0));
    CHECK_EQ(num_classes, input_blob_prob.shape(1));

    float* input_blob_prob_data = input_blob_prob.mutable_cpu_data();
    float* input_blob_label_data = input_blob_label.mutable_cpu_data();

    for (int j = 0; j < numdata; ++j) {
      for (int k = 0; k < num_classes; ++k) {
        output_blob_prob_data[j*num_classes+k] = output_blob_prob_data[j*num_classes+k] + input_blob_prob_data[j*num_classes+k] / (static_cast<float>(num_files));        
      }
      CHECK_EQ(static_cast<int>(input_blob_label_data[j]), static_cast<int>(output_blob_label_data[j])); 
    }
  }

  caffe::BlobProtoVector blob_proto_vec;
  blob_proto_vec.Clear();
  output_blob_prob.ToProto(blob_proto_vec.add_blobs());
  output_blob_label.ToProto(blob_proto_vec.add_blobs()); 
  //blob_proto_vec.SerializeToString(&output);
  WriteProtoToBinaryFile(blob_proto_vec, FLAGS_outfile);

  return 0; 
}
