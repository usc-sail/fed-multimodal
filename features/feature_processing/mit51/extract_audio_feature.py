# Author: Tiantian Feng
# USC SAIL lab, tiantiaf@usc.edu
import glob
import pdb
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
    parser = argparse.ArgumentParser(description='Extract audio features')
    parser.add_argument(
        '--raw_data_dir',
        default='/media/data/public-data/MMAction/mit', 
        type=str,
        help='source data directory')

    parser.add_argument(
        '--output_dir', 
        default='/media/data/projects/speech-privacy/fed-multimodal/',
        type=str, 
        help='output feature directory')

    parser.add_argument(
        '--feature_type', 
        default='mfcc',
        type=str, 
        help='output feature name')

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="alpha in direchlet distribution",
    )
    
    parser.add_argument("--run_extraction", default=True, action='store_true')
    parser.add_argument('--skip_extraction', dest='run_extraction', action='store_false')
    parser.add_argument("--dataset", default="mit51")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # read args
    args = parse_args()
    alpha_str = str(args.alpha).replace('.', '')
    output_data_path = Path(args.output_dir).joinpath('feature', 'audio', args.feature_type, args.dataset, f'alpha{alpha_str}')
    Path.mkdir(output_data_path, parents=True, exist_ok=True)
    
    # initialize feature processer
    feature_manager = feature_manager(args)
        
    # fetch all files for processing
    partition_dict = feature_manager.fetch_partition(alpha=1.0)
    
    print('Reading videos from folder: ', args.raw_data_dir)
    print('Total number of videos found: ', len(partition_dict.keys()))
    
    # extract data
    for client in tqdm(partition_dict):
        data_dict = partition_dict[client]
        for idx in range(len(partition_dict[client])):
            file_path = partition_dict[client][idx][1]
            video_id, _ = osp.splitext(osp.basename(file_path))
            label_str = osp.basename(osp.dirname(file_path))
            features = feature_manager.extract_mfcc_features(audio_path=file_path, 
                                                             label_str=label_str,
                                                             frame_length=40,
                                                             frame_shift=20,
                                                             max_len=150)
            # data_dict[f'{label_str}/{video_id}'] = features
            data_dict[idx].append(features)
        # saving features
        with open(str(output_data_path.joinpath(f'{client}.pkl')), 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    """
    # saving features
    save_path = str(output_data_path.joinpath(f'feature.pkl'))
    with open(save_path, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(str(output_data_path.joinpath(f'feature.pkl')), "rb") as f: 
        data_dict = pickle.load(f)

    # save for later uses

    output_data_path = Path(args.output_dir).joinpath('feature', 'audio', args.feature_type, args.dataset, f'alpha{alpha_str}', f'fold{fold_idx+1}')
    Path.mkdir(output_data_path, parents=True, exist_ok=True)

    partition_dict = feature_manager.fetch_partition(fold_idx+1, alpha=args.alpha)
    for client in partition_dict:
        for idx in range(len(partition_dict[client])):
            features = data_dict[partition_dict[client][idx][0]]
            partition_dict[client][idx].append(features)

    # save dev
    with open(output_data_path.joinpath(f'dev.pkl'), 'wb') as handle:
        pickle.dump(partition_dict['dev'], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # save test
    with open(output_data_path.joinpath(f'test.pkl'), 'wb') as handle:
        pickle.dump(partition_dict['test'], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # save train
    partition_dict.pop('dev')
    partition_dict.pop('test')
    with open(output_data_path.joinpath(f'train.pkl'), 'wb') as handle:
        pickle.dump(partition_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    """