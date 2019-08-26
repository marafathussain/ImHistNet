#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/data/caffe/data/MICCAI19_Data4
DATA=/data/caffe/data/MICCAI19_Data4
TOOLS=/data/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/MICCAI19_train_lmdb \
  $DATA/MICCAI19_mean_slice.binaryproto

echo "Done."
