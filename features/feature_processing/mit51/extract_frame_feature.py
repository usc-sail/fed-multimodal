import os
import pdb
import pdb
import glob
import torch
import pickle
import random
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
    parser = argparse.ArgumentParser(description='Extract frame level features')
    parser.add_argument(
        '--raw_data_dir',
        default='/media/data/public-data/MMAction/mit', 
        type=str,
        help='source video directory')
    parser.add_argument(
        '--output_dir', 
        default='/media/data/projects/speech-privacy/fed-multimodal/',
        type=str, 
        help='output feature directory')
    parser.add_argument(
        '--feature_type', 
        default='mobilenet_v2',
        type=str, 
        help='output feature name')
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="alpha in direchlet distribution",
    )
    parser.add_argument("--dataset", default="mit51")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    # read args
    args = parse_args()
    alpha_str = str(args.alpha).replace('.', '')
    output_data_path = Path(args.output_dir).joinpath('feature', 'video', args.feature_type, args.dataset, f'alpha{alpha_str}')
    Path.mkdir(output_data_path, parents=True, exist_ok=True)
    
    # initialize feature processer
    feature_manager = feature_manager(args)

    # fetch all files for processing
    partition_dict = feature_manager.fetch_partition(alpha=args.alpha)
    
    print('Reading videos from folder: ', args.raw_data_dir)
    print('Total number of videos found: ', len(partition_dict.keys()))
    
    # extract data
    for client in tqdm(list(partition_dict.keys())[100:500]):
        data_dict = partition_dict[client].copy()
        split = 'validation' if client == 'test' else 'training'
        if Path.exists(output_data_path.joinpath(f'{client}.pkl')) == True: continue
        for idx in range(len(partition_dict[client])):
            file_path = partition_dict[client][idx][1]
            video_id, _ = osp.splitext(osp.basename(file_path))
            label_str = osp.basename(osp.dirname(file_path))
            features = feature_manager.extract_frame_features(video_id, label_str, max_len=8, split=split)
            
            # data_dict[f'{label_str}/{video_id}'] = features
            data_dict[idx].append(features)
        # saving features
        # pdb.set_trace()
        with open(str(output_data_path.joinpath(f'{client}.pkl')), 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
