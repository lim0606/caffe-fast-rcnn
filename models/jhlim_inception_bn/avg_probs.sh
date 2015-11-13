#!/bin/bash

CAFFE_ROOT=/home/jaehyun/github/caffe-fast-rcnn

OUTFILE=avg_probs.caffelmodel
FILELIST=/home/jaehyun/github/caffe-fast-rcnn/filelist.txt

$CAFFE_ROOT/build/tools/avg_probs -outfile $OUTFILE -filelist $FILELIST -gpu 0
