# Author: Tiantian Feng, USC SAIL lab, tiantiaf@usc.edu
import pdb
import glob
import copy
import torch
import random
import pickle
import os, sys
import argparse
import numpy as np
import os.path as osp


from tqdm import tqdm
from pathlib import Path

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
from feature_manager import feature_manager

# Define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)


def parse_args():
    parser = argparse.ArgumentParser(description='Extract acc and watch features')
    parser.add_argument(
        '--raw_data_dir',
        default='/media/data/public-data/HAR/UCI-HAR', 
        type=str,
        help='source data directory'
    )
    
    parser.add_argument(
        '--output_dir', 
        default='/media/data/projects/speech-privacy/fed-multimodal/',
        type=str, 
        help='output feature directory'
    )
    
    parser.add_argument(
        "--dataset", 
        type=str,
        default="extrasensory_watch",
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
        watch_acc_output_data_path = Path(args.output_dir).joinpath(
            'feature', 
            'watch_acc', 
            args.dataset, 
            f'fold{fold_idx}'
        )
        Path.mkdir(acc_output_data_path, parents=True, exist_ok=True)
        Path.mkdir(watch_acc_output_data_path, parents=True, exist_ok=True)
        
        # initialize feature processer
        fm = feature_manager(args)
        
        # fetch all files for processing
        partition_dict = fm.fetch_partition()
        logging.info(f'Reading data from folder: {args.raw_data_dir}')
        print(f'Total number of clients found: {len(partition_dict.keys())}')
        
        # extract data
        for client_id in partition_dict:
            # logging
            logging.info(f'Extract features for {client_id}')
            acc_dict = copy.deepcopy(partition_dict[client_id])
            watch_dict = copy.deepcopy(partition_dict[client_id])
            
            if len(acc_dict) < 100 or len(watch_dict) < 100: continue
            # iterate over keys
            for idx in tqdm(range(len(acc_dict))):
                # 0. initialize acc, gyro file path
                acc_file_path = acc_dict[idx][1]
                watch_acc_file_path = acc_dict[idx][1].replace('raw_acc', 'watch_acc')

                # 1.1 read acc data
                acc_features = np.genfromtxt(
                    str(acc_file_path), 
                    dtype=float, delimiter=' '
                )[:, 1:][::5]
                
                # 1.2 normalize acc data
                mean, std = np.mean(acc_features, axis=0), np.std(acc_features, axis=0)
                acc_features = (acc_features - mean) / (std + 1e-5)
                acc_dict[idx].append(copy.deepcopy(acc_features))
                
                # 2.1 read watch acc data
                watch_acc_features = np.genfromtxt(
                    str(watch_acc_file_path), 
                    dtype=float, 
                    delimiter=' '
                )[::5]
                # pdb.set_trace()
                if watch_acc_features.shape[1] == 4:
                    watch_acc_features = watch_acc_features[:, 1:]
                # 2.2 normalize watch acc data
                mean, std = np.mean(watch_acc_features, axis=0), np.std(watch_acc_features, axis=0)
                watch_acc_features = (watch_acc_features - mean) / (std + 1e-5)
                watch_dict[idx].append(copy.deepcopy(watch_acc_features))
            # very important: final feature output format
            # [key, idx, label, feature]
            with open(acc_output_data_path.joinpath(f'{client_id}.pkl'), 'wb') as handle:
                pickle.dump(acc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(watch_acc_output_data_path.joinpath(f'{client_id}.pkl'), 'wb') as handle:
                pickle.dump(watch_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    