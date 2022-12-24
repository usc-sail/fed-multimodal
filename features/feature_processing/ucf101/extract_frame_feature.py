
import os
import pdb
import glob
import torch
import random
import pickle
import numpy as np
import argparse, sys
import os.path as osp

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch import optim, nn
from torchvision import models, transforms

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
from feature_manager import feature_manager


def parse_args():

    # read path config files
    path_conf = dict()
    with open(str(Path(os.path.realpath(__file__)).parents[3].joinpath('system.cfg'))) as f:
        for line in f:
            key, val = line.strip().split('=')
            path_conf[key] = val.replace("\"", "")

    parser = argparse.ArgumentParser(description='Extract frame level features')
    parser.add_argument(
        '--raw_data_dir',
        default=path_conf["data_dir"],
        type=str,
        help='source video directory'
    )
    parser.add_argument(
        '--output_dir', 
        default=path_conf["output_dir"],
        type=str, 
        help='output feature directory'
    )
    parser.add_argument(
        '--feature_type', 
        default='mobilenet_v2',
        type=str, 
        help='output feature name'
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="alpha in direchlet distribution",
    )
    parser.add_argument("--dataset", default="ucf101")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    # read args
    args = parse_args()
    alpha_str = str(args.alpha).replace('.', '')
    output_data_path = Path(args.output_dir).joinpath('feature', 'video', args.feature_type, args.dataset)
    Path.mkdir(output_data_path, parents=True, exist_ok=True)
    
    # initialize feature processer
    feature_manager = feature_manager(args)
    if Path.exists(output_data_path.joinpath(f'feature.pkl')) == False:
        
        # fetch all files for processing
        partition_dict = feature_manager.fetch_partition(alpha=args.alpha)
        
        print('Reading videos from folder: ', args.raw_data_dir)
        print('Total number of videos found: ', len(partition_dict.keys()))
        
        # extract data
        data_dict = dict()
        for client in tqdm(partition_dict):
            for idx in range(len(partition_dict[client])):
                file_path = partition_dict[client][idx][1]
                video_id, _ = osp.splitext(osp.basename(file_path))
                label_str = osp.basename(osp.dirname(file_path))
                features = feature_manager.extract_frame_features(
                    video_id, 
                    label_str, 
                    max_len=10
                )
                data_dict[f'{label_str}/{video_id}'] = features
            
        # saving features
        save_path = str(output_data_path.joinpath(f'feature.pkl'))
        with open(save_path, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(str(output_data_path.joinpath(f'feature.pkl')), "rb") as f: 
        data_dict = pickle.load(f)
    
    # save for later uses
    for fold_idx in range(3):
        output_data_path = Path(args.output_dir).joinpath(
            'feature', 
            'video', 
            args.feature_type, 
            args.dataset, 
            f'alpha{alpha_str}', 
            f'fold{fold_idx+1}'
        )
        Path.mkdir(output_data_path, parents=True, exist_ok=True)
    
        partition_dict = feature_manager.fetch_partition(fold_idx+1, alpha=args.alpha)
        for client in partition_dict:
            save_dict = partition_dict[client]
            for idx in range(len(partition_dict[client])):
                features = data_dict[partition_dict[client][idx][0]]
                save_dict[idx].append(features)
            
            with open(output_data_path.joinpath(f'{client}.pkl'), 'wb') as handle:
                pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)