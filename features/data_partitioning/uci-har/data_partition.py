# Author: Tiantian Feng
# USC SAIL lab, tiantiaf@usc.edu
import json
import pickle
import re, pdb
import sys, os
import argparse
import numpy as np
import pandas as pd
import configparser
import os.path as osp
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import KFold


sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
from partition_manager import partition_manager


def data_partition(args: dict):
    
    # define partition manager
    pm = partition_manager(args)
    
    # fetch all labels
    pm.fetch_label_dict()
    
    # save the partition
    output_data_path = Path(args.output_partition_path).joinpath(args.dataset)
    Path.mkdir(output_data_path, parents=True, exist_ok=True)
        
    # data root folder
    partition_dict = dict()
    partition_dict['test'] = list()
    partition_dict['dev'] = list()
    
    for data_type in ['train', 'test']:
        data_root_path = Path(args.raw_data_dir).joinpath(args.dataset)
        subject_path = data_root_path.joinpath(data_type, 'subject_'+data_type+'.txt')
        data_path = data_root_path.joinpath(data_type, 'Inertial Signals')
        client_id_data = np.genfromtxt(str(subject_path), dtype=int)
        
        # read labels
        labels = np.genfromtxt(str(data_root_path.joinpath(data_type, 'y_'+data_type+'.txt')), dtype=float)-1
        # read acc
        acc_x = np.genfromtxt(str(data_path.joinpath('body_acc_x_'+data_type+'.txt')), dtype=float)
        acc_y = np.genfromtxt(str(data_path.joinpath('body_acc_y_'+data_type+'.txt')), dtype=float)
        acc_z = np.genfromtxt(str(data_path.joinpath('body_acc_z_'+data_type+'.txt')), dtype=float)
        # read gyro
        gyro_x = np.genfromtxt(str(data_path.joinpath('body_gyro_x_'+data_type+'.txt')), dtype=float)
        gyro_y = np.genfromtxt(str(data_path.joinpath('body_gyro_y_'+data_type+'.txt')), dtype=float)
        gyro_z = np.genfromtxt(str(data_path.joinpath('body_gyro_z_'+data_type+'.txt')), dtype=float)
        
        # return unique client ids
        client_id_list = np.unique(client_id_data)
        # data for each client
        for client_id in client_id_list:
            # return data idx belonging to the client
            client_data_idx = list(np.where(client_id_data == client_id)[0])
            client_data_idx.sort()
            
            # read all file mapping => [key, idx (aka. File path), label]
            client_data_dict = dict()
            for data_idx in client_data_idx:
                label = int(labels[data_idx])
                key = f'{client_id}/{data_idx}'
                # test case, save directly, else post processing
                if data_type == 'test': partition_dict['test'].append([str(client_id), int(data_idx), int(label)])
                else: client_data_dict[key] = [str(client_id), int(data_idx), int(label)]
            # test case do nothing
            if data_type == 'test': continue
            
            # sort keys
            train_val_keys = list(client_data_dict.keys())
            train_val_keys.sort()
            
            # split train and dev from a client for later
            train_keys, dev_keys = pm.split_train_dev(train_val_keys)
            
            # read train labels
            label_list = [client_data_dict[key][2] for key in train_keys]

            # each idx of the list contains the file list
            key_idx_clients = pm.direchlet_partition(label_list)
            for shard_idx in range(len(key_idx_clients)):
                key_idx_in_shard = key_idx_clients[shard_idx]
                partition_dict[f'{client_id}-{shard_idx}'] = [client_data_dict[train_keys[key_idx]] for key_idx in key_idx_in_shard]
            for key in dev_keys:
                partition_dict['dev'].append(client_data_dict[key])
    
    # save json
    alpha_str = str(args.alpha).replace('.', '')
    jsonString = json.dumps(partition_dict, indent=4)
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
        default=f'{path_conf["output_dir"]}/partition',
        help="Output path of speech_commands data set",
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
    
    parser.add_argument("--dataset", default="uci-har")
    args = parser.parse_args()
    
    data_partition(args)
    
    
    