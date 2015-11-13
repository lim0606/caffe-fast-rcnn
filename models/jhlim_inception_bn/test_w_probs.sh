#!/bin/bash

CAFFE_ROOT=/home/jaehyun/github/caffe-fast-rcnn

TEST_MODEL=models/jhlim_inception_bn/prob.prototxt

$CAFFE_ROOT/build/tools/test_w_probs -test_model $TEST_MODEL -gpu 0 
