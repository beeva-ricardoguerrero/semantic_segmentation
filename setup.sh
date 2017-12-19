#!/bin/bash

WORKING_DIR=~/semantic_segmentation
DATASET_DIR=$WORKING_DIR/Dataset
DATASET=VOCtrainval_11-May-2012.tar
BUCKET_URL=s3://beeva-research-lab-nvirginia/poc_ricardo_semantic_segmentation


mkdir -p $WORKING_DIR/Dataset
aws s3 cp $BUCKET_URL/DATASET $DATASET_DIR/

tar -xvf $DATASET_FOLDER/$DATASET --directory $DATASET_DIR/