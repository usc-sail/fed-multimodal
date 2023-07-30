# Author: Tiantian Feng
# USC SAIL lab, tiantiaf@usc.edu
import pdb
import glob
import copy
import torch
import random
import pickle
import os, sys
import argparse
import numpy as np
import pandas as pd
import os.path as osp

from tqdm import tqdm
from pathlib import Path

from fed_multimodal.features.feature_processing.feature_manager import FeatureManager

# Define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def parse_args():
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

    parser = argparse.ArgumentParser(description='Extract acc and gyro features')
    parser.add_argument(
        '--raw_data_dir',
        default=path_conf['data_dir'], 
        type=str,
        help='source data directory'
    )
    
    parser.add_argument(
        '--output_dir', 
        default=path_conf['output_dir'],
        type=str, 
        help='output feature directory'
    )
    
    parser.add_argument(
        "--dataset", 
        type=str,
        default="ku-har",
        help="dataset name",
    )
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Read args
    args = parse_args()

    # Iterate over folds
    for fold_idx in range(1, 6):
        acc_output_data_path = Path(args.output_dir).joinpath(
            'feature', 
            'acc', 
            args.dataset, 
            f'fold{fold_idx}'
        )
        gyro_output_data_path = Path(args.output_dir).joinpath(
            'feature', 
            'gyro', 
            args.dataset, 
            f'fold{fold_idx}'
        )
        Path.mkdir(acc_output_data_path, parents=True, exist_ok=True)
        Path.mkdir(gyro_output_data_path, parents=True, exist_ok=True)
        
        # initialize feature processer
        fm = FeatureManager(args)
        
        # fetch all files for processing
        partition_dict = fm.fetch_partition(
            fold_idx=fold_idx
        )
        logging.info(f'Reading data from folder: {args.raw_data_dir}')
        logging.info(f'Total number of clients found: {len(partition_dict.keys())}')
        
        # extract data
        for client_id in partition_dict:
            acc_dict = copy.deepcopy(partition_dict[client_id])
            gyro_dict = copy.deepcopy(partition_dict[client_id])
            # the data is too small
            if len(acc_dict) < 10: continue
            logging.info(f'Process data: {client_id}')
            # iterate over keys
            for idx in tqdm(range(len(acc_dict))):
                # 0. read_data
                file_path = acc_dict[idx][1]
                off_idx = acc_dict[idx][0]*256
                data = np.array(pd.read_csv(file_path, index_col=0, header=None))[off_idx:off_idx+128][::2]
                data = np.delete(data, 3, 1)

                # 1.1 read acc data
                acc_features = data[:, :3]
                # 1.2 normalize acc data
                mean, std = np.mean(acc_features, axis=0), np.std(acc_features, axis=0)
                acc_features = (acc_features - mean) / (std + 1e-5)
                acc_dict[idx].append(copy.deepcopy(acc_features))
                
                # 2.1 read gyro data
                gyro_features = data[:, 3:]
                # 2.2 normalize gyro data
                mean, std = np.mean(gyro_features, axis=0), np.std(gyro_features, axis=0)
                gyro_features = (gyro_features - mean) / (std + 1e-5)
                gyro_dict[idx].append(copy.deepcopy(gyro_features))
                
            # very important: final feature output format
            # [key, idx, label, feature]
            with open(acc_output_data_path.joinpath(f'{client_id}.pkl'), 'wb') as handle:
                pickle.dump(acc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(gyro_output_data_path.joinpath(f'{client_id}.pkl'), 'wb') as handle:
                pickle.dump(gyro_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    