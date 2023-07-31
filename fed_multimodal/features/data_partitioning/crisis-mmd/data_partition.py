import json
import pickle
import sys, os
import re, pdb
import argparse
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
from pathlib import Path

from fed_multimodal.features.data_partitioning.partition_manager import PartitionManager

# Define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

def remove_url(text):
    text = re.sub(r'http\S+', '', text)
    # re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    return(text)

def data_partition(args: dict):
    
    # Read arguments
    num_clients, alpha = args.num_clients, args.alpha
    
    # Define partition manager
    pm = PartitionManager(args)
    
    # Fetch all labels
    pm.fetch_label_dict() # obtaining the label dictionary 

    # get the raw csv data
    data_path = Path(args.raw_data_dir).joinpath(args.dataset, 'CrisisMMD_v2.0')
    train_csv_data = pd.read_csv(data_path.joinpath("crisismmd_datasplit_all", "task_humanitarian_text_img_train.tsv"), sep='\t')
    val_csv_data = pd.read_csv(data_path.joinpath("crisismmd_datasplit_all", "task_humanitarian_text_img_dev.tsv"), sep='\t')
    test_csv_data = pd.read_csv(data_path.joinpath("crisismmd_datasplit_all", "task_humanitarian_text_img_test.tsv"), sep='\t')

    train_data_dict, dev_data_dict, test_data_dict = dict(), dict(), dict()
    
    # train dict generate from csv data
    logging.info("Partition train data")
    for i in tqdm(np.arange(train_csv_data.shape[0])):
        train_text = remove_url(train_csv_data['tweet_text'].iloc[i]).strip()
        # print(train_text)
        train_data_dict[train_csv_data['image_id'].iloc[i]] = [
            train_csv_data['image_id'].iloc[i],
            str(Path(data_path).joinpath(train_csv_data['image'].iloc[i])),
            pm.label_dict[train_csv_data['label_image'].iloc[i]],
            train_text
        ]
        
    # val dict generate from csv data
    logging.info("Partition validation data")
    for i in tqdm(np.arange(val_csv_data.shape[0])):
        val_text=remove_url(val_csv_data['tweet_text'].iloc[i]).strip()
        dev_data_dict[val_csv_data['image_id'].iloc[i]] = [
            val_csv_data['image_id'].iloc[i],
            str(Path(data_path).joinpath(val_csv_data['image'].iloc[i])),
            pm.label_dict[val_csv_data['label_image'].iloc[i]],
            val_text
        ]
        
    #test dict generate from csv data
    logging.info("Partition test data")
    for i in tqdm(np.arange(test_csv_data.shape[0])):
        test_text=remove_url(val_csv_data['tweet_text'].iloc[i]).strip()
        test_data_dict[test_csv_data['image_id'].iloc[i]] = [
            test_csv_data['image_id'].iloc[i],
            str(Path(data_path).joinpath(test_csv_data['image'].iloc[i])),
            pm.label_dict[test_csv_data['label_image'].iloc[i]],
            test_text
        ]

    train_file_ids = list(train_data_dict.keys())
    train_file_ids.sort()
    
    file_label_list = [train_data_dict[file_id][2] for file_id in train_file_ids]
    
    # Perform split
    # file_idx_clients => [client0_file_idx: array, client1_file_idx: array, ...]
    file_idx_clients = pm.direchlet_partition(
        file_label_list,
        min_sample_size=1
    )

    # Save the partition
    output_data_path = Path(args.output_partition_path).joinpath('partition', args.dataset)
    Path.mkdir(output_data_path, parents=True, exist_ok=True)

    # Obtrain train mapping
    client_data_dict = dict()
    for client_idx in range(num_clients):
        client_data_dict[client_idx] = [train_data_dict[train_file_ids[idx]] for idx in file_idx_clients[client_idx]]
    
    # Obtrain dev and test mapping
    client_data_dict["dev"] = [dev_data_dict[file_id] for file_id in dev_data_dict]
    client_data_dict["test"] = [test_data_dict[file_id] for file_id in test_data_dict]
    alpha_str = str(args.alpha).replace('.', '')

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
        help="Raw data path of crisis-mmd data set",
    )
    
    parser.add_argument(
        "--output_partition_path",
        type=str,
        default=path_conf["output_dir"],
        help="Output path of crisis-mmd data set",
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

    parser.add_argument(
        "--dataset",
        type=str, 
        default="crisis-mmd",
        help='Dataset name.'
    )
    args = parser.parse_args()
    data_partition(args)
    