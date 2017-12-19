#!/bin/bash

WORKING_DIR=~/semantic_segmentation
DATASET_DIR=$WORKING_DIR/Dataset/Pascal_voc_2012
DATASET=VOCtrainval_11-May-2012.tar
BUCKET_URL=s3://beeva-research-lab-nvirginia/poc_ricardo_semantic_segmentation
CAFFEMODEL=fcn8s-heavy-pascal.caffemodel
CAFFEMODEL_DIR=$WORKING_DIR/External_repo/fcn.berkeleyvision.org/voc-fcn8s


mkdir -p $WORKING_DIR/Dataset
aws s3 cp $BUCKET_URL/$DATASET $DATASET_DIR/
tar -xvf $DATASET_DIR/$DATASET --directory $DATASET_DIR/

mkdir -p $WORKING_DIR/External_repo
git clone https://github.com/shelhamer/fcn.berkeleyvision.org.git $WORKING_DIR/External_repo/


# Download caffemodel
aws s3 cp $BUCKET_URL/$CAFFEMODEL $CAFFEMODEL_DIR/

mkdir WORKING_DIR=~/semantic_segmentation/Caffe/Results