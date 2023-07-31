#!/bin/bash
source ../system.cfg
echo "Data folder: "$data_dir

if [[ ! -e $data_dir ]]; then
    mkdir $data_dir
fi

cd $data_dir && mkdir crema_d
cd crema_d

git-lfs clone https://github.com/CheyneyComputerScience/CREMA-D.git