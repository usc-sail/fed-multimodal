#!/bin/bash
source ../system.cfg
echo "Data folder: "$data_dir

if [[ ! -e $data_dir ]]; then
    mkdir $data_dir
fi

cd $data_dir && mkdir ptb-xl
cd ptb-xl

wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
