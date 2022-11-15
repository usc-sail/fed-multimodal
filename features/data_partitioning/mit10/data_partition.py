# Author: Tiantian Feng, USC SAIL lab, tiantiaf@usc.edu
import sys, os
import pickle
import re, pdb
import argparse
import torchaudio
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import KFold


sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
from partition_manager import partition_manager


def data_partition(args: dict):
    
    # Read arguments
    num_clients, alpha = args.num_clients, args.alpha
    
    # Define partition manager
    pm = partition_manager(args)
    
    # Fetch all files for processing
    pm.fetch_filelist()
    
    # Fetch all labels
    pm.fetch_label_dict()
    
    # Creating dictionary for partition
    # Since there is no test folder in raw data, we use dev as test
    # We split 20% data from train as dev
    train_data_dict, test_data_dict = dict(), dict()
    # Iterate over train file list
    # Create train_data_dict => {key: [key, file_path, label]}
    for file_path in tqdm(pm.train_file_list):
        video_id, _ = osp.splitext(osp.basename(file_path))
        label_str = osp.basename(osp.dirname(file_path))
        if label_str not in pm.label_dict: continue
        if Path.exists(Path(args.raw_data_dir).joinpath('rawframes', 'training', label_str, video_id, 'img_00071.jpg')) == False: continue
        train_data_dict[f'{label_str}/{video_id}'] = [f'{label_str}/{video_id}', file_path, pm.label_dict[label_str]]
    # Iterate over dev file list
    # Create test_data_dict => {key: [key, file_path, label]}
    for file_path in tqdm(pm.test_file_list):
        video_id, _ = osp.splitext(osp.basename(file_path))
        label_str = osp.basename(osp.dirname(file_path))
        if label_str not in pm.label_dict: continue
        if Path.exists(Path(args.raw_data_dir).joinpath('rawframes', 'validation', label_str, video_id, 'img_00071.jpg')) == False: continue
        test_data_dict[f'{label_str}/{video_id}'] = [f'{label_str}/{video_id}', file_path, pm.label_dict[label_str]]
    
    # Read keys, and sort, so we have the same keys in order
    train_val_file_id = list(train_data_dict.keys())
    test_file_id = list(test_data_dict.keys())
    train_val_file_id.sort()
    test_file_id.sort()
    
    # Split train and dev
    train_file_id, dev_file_id = pm.split_train_dev(train_val_file_id)
    
    # Read labels from train files
    file_label_list = [train_data_dict[file_id][2] for file_id in train_file_id]

    # Perform split
    # file_idx_clients => [client0_file_idx: array, client1_file_idx: array, ...]
    file_idx_clients = pm.direchlet_partition(file_label_list)

    # Save the partition
    output_data_path = Path(args.output_partition_path).joinpath(args.dataset)
    Path.mkdir(output_data_path, parents=True, exist_ok=True)

    # Obtrain train mapping
    client_data_dict = dict()
    for client_idx in range(num_clients):
        client_data_dict[client_idx] = [train_data_dict[train_file_id[idx]] for idx in file_idx_clients[client_idx]]
    
    # Obtrain dev and test mapping
    client_data_dict["dev"] = [train_data_dict[file_id] for file_id in dev_file_id]
    client_data_dict["test"] = [test_data_dict[file_id] for file_id in test_file_id]
    alpha_str = str(args.alpha).replace('.', '')
    with open(output_data_path.joinpath(f'partition_alpha{alpha_str}.pkl'), 'wb') as handle:
        pickle.dump(client_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default="/media/data/public-data/MMAction/mit",
        help="Raw data path of Moments In Time dataset",
    )
    
    parser.add_argument(
        "--output_partition_path",
        type=str,
        default="/media/data/projects/speech-privacy/fed-multimodal/partition",
        help="Output path of speech_commands data set",
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
        default=10, 
        help='Number of clients to cut from whole data.'
    )
    parser.add_argument(
        "--dataset",
        type=str, 
        default="mit10",
        help='Dataset name.'
    )
    args = parser.parse_args()
    data_partition(args)
    
    
    