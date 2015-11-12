#!/bin/bash

CAFFE_ROOT=/home/jaehyun/github/caffe-fast-rcnn

TEST_MODEL=models/jhlim_inception_bn/train_val.prototxt
#WEIGHT=models/jhlim_inception_bn/inception_bn_stepsize_6400_iter_1200000.caffemodel
WEIGHT=/media/data1/pretrained_nets/inception_bn_ilsvrc12/shuni/inception_bn_stepsize_6400_iter_1200000.caffemodel
LABEL_LIST=/media/data1/image/ilsvrc12/labellist.txt
OUTFILE=inception_bn_prob.txt
TARGET_BLOB_NAME=loss3/prob
SAVEFOLDER=shuni

$CAFFE_ROOT/build/tools/predict_bn_w_saved_blobs -test_model $TEST_MODEL -weights $WEIGHT -labellist $LABEL_LIST -outfile $OUTFILE -target_blob $TARGET_BLOB_NAME -savefolder $SAVEFOLDER -gpu 0 