# Author: Tiantian Feng
# USC SAIL lab, tiantiaf@usc.edu
import json
import sys, os
import re, pdb
import argparse
from pathlib import Path


from fed_multimodal.features.data_partitioning.partition_manager import PartitionManager


def data_partition(args: dict):
    
    # Read arguments
    num_clients, alpha = args.num_clients, args.alpha
    
    # Define partition manager
    pm = PartitionManager(args)
    
    # Fetch all labels
    pm.fetch_label_dict()
    
    # read train, dev, and test dict
    train_data_dict, dev_data_dict, test_data_dict = dict(), dict(), dict()
    with open(Path(args.raw_data_dir).joinpath(args.dataset, "train.jsonl"), "r") as f:
        for line in f:
            line_data = json.loads(line)
            train_data_dict[line_data['img']] = [
                line_data['img'], 
                str(Path(args.raw_data_dir).joinpath(args.dataset, line_data['img'])), 
                line_data['label'],
                line_data['text']
            ]
    with open(Path(args.raw_data_dir).joinpath(args.dataset, "dev_seen.jsonl"), "r") as f:
        for line in f:
            line_data = json.loads(line)
            dev_data_dict[line_data['img']] = [
                line_data['img'], 
                str(Path(args.raw_data_dir).joinpath(args.dataset, line_data['img'])), 
                line_data['label'],
                line_data['text']
            ]
    with open(Path(args.raw_data_dir).joinpath(args.dataset, "test_seen.jsonl"), "r") as f: 
        for line in f:
            line_data = json.loads(line)
            test_data_dict[line_data['img']] = [
                line_data['img'], 
                str(Path(args.raw_data_dir).joinpath(args.dataset, line_data['img'])), 
                line_data['label'],
                line_data['text']
            ]
    
    # Creating dictionary for partition
    # Read labels from train files
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
    output_data_path = Path(args.output_partition_path).joinpath("partition", args.dataset)
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
        help="Raw data path of hateful memes data set",
    )
    
    parser.add_argument(
        "--output_partition_path",
        type=str,
        default=path_conf["output_dir"],
        help="Output path of hateful memes data set",
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
        default=40, 
        help='Number of clients to cut from whole data.'
    )
    parser.add_argument(
        "--dataset",
        type=str, 
        default="hateful_memes",
        help='Dataset name.'
    )
    args = parser.parse_args()
    data_partition(args)
    
    
    