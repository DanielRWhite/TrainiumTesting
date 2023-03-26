#!/bin/bash
#sudo apt update && sudo apt install -y aria2

DATASET_DIR=$(pwd)/dataset

mkdir -p $(pwd)/dataset/.archive/{train,val}
mkdir -p $(pwd)/dataset/{train,val}

echo "Downloading training dataset..."
aria2c -i dataset_links.train.txt -c 16 -x 16 -d $(pwd)/dataset/.archive/train

echo "Downloading validation dataset..."
aria2c -i dataset_links.val.txt -c 16 -x 16 -d $(pwd)/dataset/.archive/val

echo "Extracting training dataset..."
cd $DATASET_DIR/.archive/train
unzip -q \*.zip -d $DATASET_DIR/train

echo "Extracting validation dataset..."
cd $DATASET_DIR/.archive/val
unzip -q \*.zip -d $DATASET_DIR/val

echo "Done!"