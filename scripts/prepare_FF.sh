#!/bin/bash

if [ ! -f ".project-root" ]; then
  echo "Please run this script from the root of the project"
  exit 1
fi

DATASET_DIR=datasets/FF
CONFIG_DIR=config/datasets/FF

mkdir -p $DATASET_DIR
mkdir -p $CONFIG_DIR/test

find $DATASET_DIR/DF/* -type f | sort > $CONFIG_DIR/test/DF.txt
find $DATASET_DIR/F2F/* -type f | sort > $CONFIG_DIR/test/F2F.txt
find $DATASET_DIR/FS/* -type f | sort > $CONFIG_DIR/test/FS.txt
find $DATASET_DIR/NT/* -type f | sort > $CONFIG_DIR/test/NT.txt
find $DATASET_DIR/real/* -type f | sort > $CONFIG_DIR/test/real.txt
