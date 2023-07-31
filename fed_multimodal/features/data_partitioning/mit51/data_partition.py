# Author: Tiantian Feng, USC SAIL lab, tiantiaf@usc.edu
import json
import sys, os
import re, pdb
import argparse
import os.path as osp
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import KFold

from fed_multimodal.features.data_partitioning.partition_manager import PartitionManager


def data_partition(args: dict):
    
    # Read arguments
    num_clients, alpha = args.num_clients, args.alpha
    
    # Define partition manager
    pm = PartitionManager(args)
    
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

    # step 0 train data split
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=path_conf["data_dir"],
        help="Raw data path of mit51 data set",
    )
    
    parser.add_argument(
        "--output_partition_path",
        type=str,
        default=path_conf["output_dir"],
        help="Output path of mit51 data set",
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
        default=1000, 
        help='Number of clients to cut from whole data.'
    )
    parser.add_argument(
        "--dataset",
        type=str, 
        default="mit51",
        help='Dataset name.'
    )
    args = parser.parse_args()
    
    data_partition(args)
    
    
    