#!/bin/bash
source ../system.cfg
echo "Data folder: "$data_dir

if [[ ! -e $data_dir ]]; then
    mkdir $data_dir
fi

cd $data_dir 

if [[ ! -e ptb-xl ]]; then
    mkdir ptb-xl
fi

cd ptb-xl

wget -r -N -c -np --no-check-certificate https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip
mv physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip

# unzip
unzip ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip
