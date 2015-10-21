#!/usr/bin/env sh

./build/tools/caffe train -solver models/jhlim_inception_bn/solver_stepsize_6400.prototxt -gpu 0,1
#./build/tools/caffe train -solver models/jhlim_inception_bn/solver.prototxt -snapshot models/jhlim_inception_bn/inception_bn_solver_stepsize_6400_iter_1200000.solverstate -gpu 0

