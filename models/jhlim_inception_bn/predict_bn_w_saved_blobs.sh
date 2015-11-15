#!/bin/bash

CAFFE_ROOT=/home/jaehyun/github/caffe-fast-rcnn

TEST_MODEL=models/jhlim_inception_bn/train_val.prototxt
#WEIGHT=models/jhlim_inception_bn/inception_bn_stepsize_6400_iter_1200000.caffemodel
LABEL_LIST=/media/data1/image/ilsvrc12/labellist.txt
TARGET_BLOB_NAME=loss3/prob


# shuni
SAVEFOLDER=shuni
OUTFILE=tmp/$SAVEFOLDER"_inception_bn_prob.caffemodel"
WEIGHT=/media/data1/pretrained_nets/inception_bn_ilsvrc12/$SAVEFOLDER/inception_bn_stepsize_6400_iter_1200000.caffemodel

$CAFFE_ROOT/build/tools/predict_bn_w_saved_blobs -test_model $TEST_MODEL -weights $WEIGHT -labellist $LABEL_LIST -outfile $OUTFILE -target_blob $TARGET_BLOB_NAME -savefolder tmp/$SAVEFOLDER -gpu 0


# jhl
SAVEFOLDER=jhl
OUTFILE=tmp/$SAVEFOLDER"_inception_bn_prob.caffemodel"
WEIGHT=/media/data1/pretrained_nets/inception_bn_ilsvrc12/$SAVEFOLDER/inception_bn_stepsize_6400_iter_1200000.caffemodel

$CAFFE_ROOT/build/tools/predict_bn_w_saved_blobs -test_model $TEST_MODEL -weights $WEIGHT -labellist $LABEL_LIST -outfile $OUTFILE -target_blob $TARGET_BLOB_NAME -savefolder tmp/$SAVEFOLDER -gpu 0


# jhl2
SAVEFOLDER=jhl2
OUTFILE=tmp/$SAVEFOLDER"_inception_bn_prob.caffemodel"
WEIGHT=/media/data1/pretrained_nets/inception_bn_ilsvrc12/$SAVEFOLDER/inception_bn_stepsize_6400_iter_1200000.caffemodel

$CAFFE_ROOT/build/tools/predict_bn_w_saved_blobs -test_model $TEST_MODEL -weights $WEIGHT -labellist $LABEL_LIST -outfile $OUTFILE -target_blob $TARGET_BLOB_NAME -savefolder tmp/$SAVEFOLDER -gpu 0


# old1
SAVEFOLDER=old1
OUTFILE=tmp/$SAVEFOLDER"_inception_bn_prob.caffemodel"
WEIGHT=/media/data1/pretrained_nets/inception_bn_ilsvrc12/$SAVEFOLDER/inception_bn_solver_stepsize_6400_iter_1200000.caffemodel

$CAFFE_ROOT/build/tools/predict_bn_w_saved_blobs -test_model $TEST_MODEL -weights $WEIGHT -labellist $LABEL_LIST -outfile $OUTFILE -target_blob $TARGET_BLOB_NAME -savefolder tmp/$SAVEFOLDER -gpu 0

# old2
SAVEFOLDER=old2
OUTFILE=tmp/$SAVEFOLDER"_inception_bn_prob.caffemodel"
WEIGHT=/media/data1/pretrained_nets/inception_bn_ilsvrc12/$SAVEFOLDER/inception_bn_solver4_iter_2400000.caffemodel

$CAFFE_ROOT/build/tools/predict_bn_w_saved_blobs -test_model $TEST_MODEL -weights $WEIGHT -labellist $LABEL_LIST -outfile $OUTFILE -target_blob $TARGET_BLOB_NAME -savefolder tmp/$SAVEFOLDER -gpu 0 
