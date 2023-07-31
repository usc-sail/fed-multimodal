# Author: Tiantian Feng, USC SAIL lab, tiantiaf@usc.edu
import json
import sys, os
import re, pdb
import argparse
import numpy as np
import os.path as osp
from pathlib import Path

train_file_list = ['ucf101_train_split_1_rawframes.txt', 'ucf101_train_split_2_rawframes.txt', 'ucf101_train_split_3_rawframes.txt']
test_file_list = ['ucf101_val_split_1_rawframes.txt', 'ucf101_val_split_2_rawframes.txt', 'ucf101_val_split_3_rawframes.txt']

from fed_multimodal.features.data_partitioning.partition_manager import PartitionManager


def data_partition(args: dict):
    
    # read arguments
    num_clients, alpha = args.num_clients, args.alpha
    
    # define partition manager
    pm = PartitionManager(args)
    
    # fetch all files for processing
    pm.fetch_filelist()
    
    # fetch all labels
    pm.fetch_label_dict()
    
    # creating dictionary for partition
    data_dict = dict()
    for file_path in pm.file_list:
        video_id, _ = osp.splitext(osp.basename(file_path))
        label_str = osp.basename(osp.dirname(file_path))
        data_dict[f'{label_str}/{video_id}'] = [f'{label_str}/{video_id}', file_path, pm.label_dict[label_str]]
    file_id_list = list(data_dict.keys())
    
    # iterate over folds
    for fold_idx in range(len(train_file_list)):
    
        # read train and test split
        with open(Path(args.raw_data_dir).joinpath(args.dataset, train_file_list[fold_idx])) as f: train_split_files = f.readlines()
        with open(Path(args.raw_data_dir).joinpath(args.dataset, test_file_list[fold_idx])) as f: test_split_files = f.readlines()
        
        # split train and validation
        train_val_file_id, test_file_id = list(), list()
        for line in train_split_files:
            file_id = line.split(' ')[0]
            if file_id in file_id_list: train_val_file_id.append(file_id)
        for line in test_split_files:
            file_id = line.split(' ')[0]
            if file_id in file_id_list: test_file_id.append(file_id)
        
        train_val_file_id.sort()
        test_file_id.sort()

        # split train and dev
        train_file_id, dev_file_id = pm.split_train_dev(train_val_file_id)
        
        # read labels
        file_label_list = [data_dict[file_id][2] for file_id in train_file_id]
        
        # each idx of the list contains the file list
        file_idx_clients = pm.direchlet_partition(file_label_list)
        client_label_list = list()
        for client_idx in range(num_clients):
            lable_list = list()
            for idx in file_idx_clients[client_idx]:
                lable_list.append(data_dict[train_file_id[idx]][2])
            client_label_list.append(len(np.unique(lable_list)))
    
        # save the partition
        output_data_path = Path(args.output_partition_path).joinpath(
            'partition', 
            args.dataset, 
            f'fold{fold_idx+1}'
        )
        Path.mkdir(output_data_path, parents=True, exist_ok=True)
        
        client_data_dict = dict()
        for client_idx in range(num_clients):
            client_data_dict[client_idx] = [data_dict[train_file_id[idx]] for idx in file_idx_clients[client_idx]]
        
        client_data_dict["dev"] = [data_dict[file_id] for file_id in dev_file_id]
        client_data_dict["test"] = [data_dict[file_id] for file_id in test_file_id]
        alpha_str = str(args.alpha).replace('.', '')
        
        # dump to json
        jsonString = json.dumps(client_data_dict, indent=4)
        jsonFile = open(str(output_data_path.joinpath(f'partition_alpha{alpha_str}.json')), "w")
        jsonFile.write(jsonString)
        jsonFile.close()
        
        
if __name__ == "__main__":
    
    # read path config files
    path_conf = dict()
    with open(str(Path(os.path.realpath(__file__)).parents[3].joinpath('system.cfg'))) as f:
        for line in f:
            key, val = line.strip().split('=')
            path_conf[key] = val.replace("\"", "")
            
    # If default setting
    if path_conf["data_dir"] == ".":
        path_conf["data_dir"] = str(Path(os.path.realpath(__file__)).parents[3].joinpath('data'))
    if path_conf["output_dir"] == ".":
        path_conf["output_dir"] = str(Path(os.path.realpath(__file__)).parents[3].joinpath('output'))
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=path_conf["data_dir"],
        help="Raw data path of ucf101 data set",
    )
    
    parser.add_argument(
        "--output_partition_path",
        type=str,
        default=path_conf["output_dir"],
        help="Output path of ucf101 data set",
    )

    parser.add_argument(
        "--setup",
        type=str,
        default="federated",
        help="setup of the experiment: centralized/federated",
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="alpha in direchlet distribution",
    )
    
    parser.add_argument(
        '--num_clients', 
        type=int,
        default=100,
        help='Number of clients to cut from whole data.'
    )
    parser.add_argument("--dataset", default="ucf101")
    args = parser.parse_args()
    data_partition(args)
    
    
    