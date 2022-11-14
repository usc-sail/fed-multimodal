# Author: Tiantian Feng, USC SAIL lab, tiantiaf@usc.edu
import pdb
import glob
import torch
import random
import pickle
import os, sys
import argparse
import opensmile
import torchaudio
import numpy as np
import os.path as osp


from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
from feature_manager import feature_manager


def parse_args():
    parser = argparse.ArgumentParser(description='Extract acc and gyro features')
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
        "--alpha",
        type=float,
        default=1.0,
        help="alpha in direchlet distribution",
    )
    
    parser.add_argument(
        "--dataset", 
        type=str,
        default="uci-har",
        help="data set name",
    )
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # read args
    args = parse_args()
    alpha_str = str(args.alpha).replace('.', '')
    acc_output_data_path = Path(args.output_dir).joinpath('feature', 'acc', args.dataset, f'alpha{alpha_str}')
    gyro_output_data_path = Path(args.output_dir).joinpath('feature', 'gyro', args.dataset, f'alpha{alpha_str}')
    Path.mkdir(acc_output_data_path, parents=True, exist_ok=True)
    Path.mkdir(gyro_output_data_path, parents=True, exist_ok=True)
    
    # initialize feature processer
    fm = feature_manager(args)
    
    # fetch all files for processing
    partition_dict = fm.fetch_partition(alpha=args.alpha)
    acc_dict = partition_dict.copy()
    gyro_dict = partition_dict.copy()
    
    print('Reading data from folder: ', args.raw_data_dir)
    print('Total number of clients found: ', len(partition_dict.keys()))
    
    for data_type in ['train', 'test']:
        data_root_path = Path(args.raw_data_dir)
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
    
        # extract data
        for client_id in acc_dict:
            # skip condition
            if client_id == 'test' and data_type != 'test': continue
            if data_type == 'test' and client_id != 'test': continue
            # iterate over keys
            for idx in range(len(acc_dict[client_id])):
                # pdb.set_trace()
                # 0. initialize acc, gyro data
                acc_features, gyro_features = np.zeros([128, 3]), np.zeros([128, 3])
                # 1.1 read acc data
                acc_features[:, 0] = acc_x[idx, :]
                acc_features[:, 1] = acc_y[idx, :]
                acc_features[:, 2] = acc_z[idx, :]
                # 1.2 normalize acc data
                mean, std = np.mean(acc_features, axis=0), np.std(acc_features, axis=0)
                acc_features = (acc_features - mean) / (std + 1e-5)
                acc_dict[client_id][idx].append(acc_features)
                
                # 2.1 read gyro data
                gyro_features[:, 0] = gyro_x[idx, :]
                gyro_features[:, 1] = gyro_y[idx, :]
                gyro_features[:, 2] = gyro_z[idx, :]
                # 2.2 normalize gyro data
                mean, std = np.mean(gyro_features, axis=0), np.std(gyro_features, axis=0)
                gyro_features = (gyro_features - mean) / (std + 1e-5)
                gyro_dict[client_id][idx].append(gyro_features)
            # very important: final feature output format
            # [key, idx, label, feature]
            with open(acc_output_data_path.joinpath(f'{client_id}.pkl'), 'wb') as handle:
                pickle.dump(acc_dict[client_id], handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(gyro_output_data_path.joinpath(f'{client_id}.pkl'), 'wb') as handle:
                pickle.dump(gyro_dict[client_id], handle, protocol=pickle.HIGHEST_PROTOCOL)
    