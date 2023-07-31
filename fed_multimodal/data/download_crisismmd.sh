#!/bin/bash
source ../system.cfg
echo "Data folder: "$data_dir

if [[ ! -e $data_dir ]]; then
    mkdir $data_dir
fi

cd $data_dir 

if [[ ! -e crisis-mmd ]]; then
    mkdir crisis-mmd
fi

cd crisis-mmd

wget -r -N -c -np --no-check-certificate https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz
mv crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz CrisisMMD_v2.0.tar.gz

tar -xvzf CrisisMMD_v2.0.tar.gz && cd CrisisMMD_v2.0/
unzip crisismmd_datasplit_all.zip

cd ..
rm -r crisisnlp.qcri.org

