#!/bin/bash

CAFFE_ROOT=/home/jaehyun/github/caffe-fast-rcnn

TRAIN_MODEL=models/jhlim_inception_bn/train_val.prototxt
TEST_MODEL=models/jhlim_inception_bn/train_val.prototxt
#WEIGHT=models/jhlim_inception_bn/inception_bn_stepsize_6400_iter_1200000.caffemodel


# shuni
SAVEFOLDER=shuni
WEIGHT=/media/data1/pretrained_nets/inception_bn_ilsvrc12/$SAVEFOLDER/inception_bn_stepsize_6400_iter_1200000.caffemodel

$CAFFE_ROOT/build/tools/save_bn -train_model $TRAIN_MODEL -weights $WEIGHT -train_iterations 1000 -savefolder tmp/$SAVEFOLDER -gpu 0


# jhl
SAVEFOLDER=jhl
WEIGHT=/media/data1/pretrained_nets/inception_bn_ilsvrc12/$SAVEFOLDER/inception_bn_stepsize_6400_iter_1200000.caffemodel

$CAFFE_ROOT/build/tools/save_bn -train_model $TRAIN_MODEL -weights $WEIGHT -train_iterations 1000 -savefolder tmp/$SAVEFOLDER -gpu 0


# jhl2
SAVEFOLDER=jhl2
WEIGHT=/media/data1/pretrained_nets/inception_bn_ilsvrc12/$SAVEFOLDER/inception_bn_stepsize_6400_iter_1200000.caffemodel

$CAFFE_ROOT/build/tools/save_bn -train_model $TRAIN_MODEL -weights $WEIGHT -train_iterations 1000 -savefolder tmp/$SAVEFOLDER -gpu 0


# old1
SAVEFOLDER=old1
WEIGHT=/media/data1/pretrained_nets/inception_bn_ilsvrc12/$SAVEFOLDER/inception_bn_solver_stepsize_6400_iter_1200000.caffemodel

$CAFFE_ROOT/build/tools/save_bn -train_model $TRAIN_MODEL -weights $WEIGHT -train_iterations 1000 -savefolder tmp/$SAVEFOLDER -gpu 0


# old2
SAVEFOLDER=old2
WEIGHT=/media/data1/pretrained_nets/inception_bn_ilsvrc12/$SAVEFOLDER/inception_bn_solver4_iter_2400000.caffemodel

$CAFFE_ROOT/build/tools/save_bn -train_model $TRAIN_MODEL -weights $WEIGHT -train_iterations 1000 -savefolder tmp/$SAVEFOLDER -gpu 0
 
