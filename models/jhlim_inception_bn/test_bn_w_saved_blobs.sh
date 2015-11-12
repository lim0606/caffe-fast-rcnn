#!/bin/bash

CAFFE_ROOT=/home/jaehyun/github/caffe-fast-rcnn

TEST_MODEL=models/jhlim_inception_bn/train_val.prototxt
#WEIGHT=models/jhlim_inception_bn/inception_bn_stepsize_6400_iter_1200000.caffemodel
WEIGHT=/media/data1/pretrained_nets/inception_bn_ilsvrc12/shuni/inception_bn_stepsize_6400_iter_1200000.caffemodel
SAVEFOLDER=shuni

$CAFFE_ROOT/build/tools/test_bn_w_saved_blobs -test_model $TEST_MODEL -weights $WEIGHT -savefolder $SAVEFOLDER -gpu 0 
