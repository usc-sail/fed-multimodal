#!/bin/bash
source ../system.cfg
echo "Data folder: "$data_dir

if [[ ! -e $data_dir ]]; then
    mkdir $data_dir
fi

cd $data_dir && mkdir meld
cd meld

#!/bin/bash
if [[ ! -e "MELD.Raw.tar.gz" ]]; then
    wget -N http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz
fi

echo =========Begin Extracting MELD.RAW.tar.gz==================
tar -x -f MELD.Raw.tar.gz
echo =========Done Extracting MELD.RAW.tar.gz===================

cd MELD.Raw
echo =========Begin Extracting MELD.Raw/train.tar.gz============
tar -x -f train.tar.gz
echo =========Done Extracting MELD.Raw/train.tar.gz=============

echo =========Begin Extracting MELD.Raw/dev.tar.gz==============
tar -x -f dev.tar.gz
echo =========Done Extracting MELD.Raw/dev.tar.gz================

echo =========Begin Extracting MELD.Raw/test.tar.gz==============
tar -x -f test.tar.gz
echo =========Done Extracting MELD.Raw/test.tar.gz===============


cd train_splits
NUM_FILES=$(find ./ -name "*.mp4" | wc -l)
mkdir -p "waves"
for i in *.mp4; do
    ffmpeg -hide_banner -loglevel error -y -i "$i" "./waves/$(basename "$i" .mp4).wav"
  echo "$i"
done | pv -l -s "$NUM_FILES" >/dev/null

cd ../output_repeated_splits_test
NUM_FILES=$(find ./ -name "*.mp4" | wc -l)
mkdir -p "waves"
for i in *.mp4; do
    ffmpeg -hide_banner -loglevel error -y -i "$i" "./waves/$(basename "$i" .mp4).wav"
  echo "$i"
done | pv -l -s "$NUM_FILES" >/dev/null

cd ../dev_splits_complete
NUM_FILES=$(find ./ -name "*.mp4" | wc -l)
mkdir -p "waves"
for i in *.mp4; do
    ffmpeg -hide_banner -loglevel error -y -i "$i" "./waves/$(basename "$i" .mp4).wav"
  echo "$i"
done | pv -l -s "$NUM_FILES" >/dev/null


