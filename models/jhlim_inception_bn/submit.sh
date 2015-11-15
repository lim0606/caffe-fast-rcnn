#!/bin/bash

CAFFE_ROOT=/home/jaehyun/github/caffe-fast-rcnn

TARGET_BLOB_NAME=loss3/argmax
TARGET_LAYER_NAME=loss3/argmax

# top1
OUTFILE=tmp/submit_top1_w_12label.txt
TEST_MODEL=models/jhlim_inception_bn/submit_top1.prototxt

$CAFFE_ROOT/build/tools/submit -test_model $TEST_MODEL -outfile $OUTFILE -target_blob $TARGET_BLOB_NAME -target_layer $TARGET_LAYER_NAME -gpu 0

# top5
OUTFILE=tmp/submit_top5_w_12label.txt
TEST_MODEL=models/jhlim_inception_bn/submit_top5.prototxt

$CAFFE_ROOT/build/tools/submit -test_model $TEST_MODEL -outfile $OUTFILE -target_blob $TARGET_BLOB_NAME -target_layer $TARGET_LAYER_NAME -gpu 0


