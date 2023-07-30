# Author: Tiantian Feng
# USC SAIL lab, tiantiaf@usc.edu
import json
import copy
import pickle
import re, pdb
import sys, os
import argparse
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import KFold


from fed_multimodal.features.data_partitioning.partition_manager import PartitionManager


def data_partition(args: dict):
    
    # define partition manager
    pm = PartitionManager(args)
    
    # fetch all labels
    pm.fetch_label_dict()
    
    # save the partition
    output_data_path = Path(args.output_partition_path).joinpath('partition', args.dataset)
    Path.mkdir(output_data_path, parents=True, exist_ok=True)
        
    # data root folder
    partition_dict = dict()
    partition_dict['test'] = list()
    partition_dict['dev'] = list()

    # read all keys
    data_root_path = Path(args.raw_data_dir).joinpath(args.dataset)
    client_data_dict = dict()
    for label in pm.label_dict:
        for file_path in os.listdir(data_root_path.joinpath('Trimmed_interpolated_data', label)):
            # read client id
            file_full_path = data_root_path.joinpath('Trimmed_interpolated_data', label, file_path)
            client_id = file_path.split('_')[0]
            if client_id == '1101': continue
            if client_id not in client_data_dict:
                client_data_dict[client_id] = dict()
            # read data
            data_df = np.array(pd.read_csv(file_full_path, index_col=0, header=None))[::2]
            n_data = int(len(data_df) / 128)
            for idx in range(n_data):
                client_data_dict[client_id][f'{label}/{file_path}-{idx}'] = [idx, str(file_full_path), pm.label_dict[label]]
                
    # train data
    client_keys = list(client_data_dict.keys())
    client_keys.sort()
    
    # create 5 fold
    kf = KFold(
        n_splits=5, 
        random_state=None, 
        shuffle=False
    )
    
    # extract partition for each fold
    for fold_idx, split_idx in enumerate(kf.split(client_keys)):
        # save partition dictionary
        partition_dict = dict()
        partition_dict['dev'] = list()
        partition_dict['test'] = list()
        # save data path
        output_data_path = Path(args.output_partition_path).joinpath(
            'partition',
            args.dataset, 
            f'fold{fold_idx+1}'
        )
        Path.mkdir(
            output_data_path, 
            parents=True, 
            exist_ok=True
        )

        # train clients, test clients
        train_idx, test_idx = split_idx
        train_clients = [client_keys[idx] for idx in train_idx]
        test_clients = [client_keys[idx] for idx in test_idx]
        
        # iterate clients
        for client_id in train_clients:
            # read client data
            client_dict = copy.deepcopy(client_data_dict[client_id])
            # read client keys
            train_dev_file_id = list(client_dict.keys())
            train_dev_file_id.sort()
            
            # partition train/dev keys
            train_file_id, dev_file_id = pm.split_train_dev(train_dev_file_id)
            # copy to partition data
            partition_dict[client_id] = list()
            for file_id in train_file_id:
                partition_dict[client_id].append(client_data_dict[client_id][file_id])
            for file_id in dev_file_id:
                partition_dict['dev'].append(client_data_dict[client_id][file_id])
        
        # save test files
        for client_id in test_clients:
            # read client data
            client_dict = copy.deepcopy(client_data_dict[client_id])
            # read client keys
            test_file_id = list(client_dict.keys())
            test_file_id.sort()
            # copy to partition data
            for file_id in test_file_id:
                partition_dict['test'].append(client_data_dict[client_id][file_id])
        
        # dump the dictionary
        jsonString = json.dumps(partition_dict, indent=4)
        jsonFile = open(str(output_data_path.joinpath(f'partition.json')), "w")
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
        help="Raw data path of extrasensory data set",
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="alpha in direchlet distribution",
    )
    
    parser.add_argument(
        "--output_partition_path",
        type=str,
        default=path_conf["output_dir"],
        help="Output path of ku-har data set",
    )
    
    parser.add_argument(
        '--num_clients', 
        type=int, 
        default=10, 
        help='Number of shards to split a subject data.'
    )

    parser.add_argument(
        "--setup",
        type=str,
        default="federated",
        help="setup of the experiment: centralized/federated",
    )
    
    parser.add_argument("--dataset", default="ku-har")
    args = parser.parse_args()
    
    data_partition(args)
    
    
    