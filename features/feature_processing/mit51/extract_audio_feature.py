# Author: Tiantian Feng
# USC SAIL lab, tiantiaf@usc.edu
import pdb
import glob
import torch
import random
import pickle
import os, sys
import logging
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
    partition_dict = feature_manager.fetch_partition(alpha=args.alpha)
    print('Reading videos from folder: ', args.raw_data_dir)
    print('Total number of videos found: ', len(partition_dict.keys()))
    
    # extract data
    base_data_path = Path(args.output_dir).joinpath('feature', 'audio', args.feature_type, args.dataset, f'alpha10')
    client_file_paths = os.listdir(base_data_path)
    client_file_paths.sort()
    if len(client_file_paths) != len(partition_dict) and args.alpha == 1.0:
        for client in tqdm(partition_dict):
            data_dict = partition_dict[client].copy()
            if Path.exists(output_data_path.joinpath(f'{client}.pkl')) == True: continue
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
    

    if len(client_file_paths) == len(partition_dict) and args.alpha != 1.0:
        train_dict = dict()
        logging.info('Read alpha=1.0 data')
        for client_file_path in tqdm(client_file_paths[:-2]):
            with open(str(base_data_path.joinpath(client_file_path)), "rb") as f: 
                client_data = pickle.load(f)
            for idx in range(len(client_data)):
                key = client_data[idx][0]
                train_dict[key] = client_data[idx]

        client_ids = list(partition_dict.keys())
        logging.info(f'Save alpha={args.alpha} data')
        for client_id in tqdm(client_ids[:-2]):
            save_data = list()
            for idx in range(len(partition_dict[client_id])):
                key = partition_dict[client_id][idx][0]
                save_data.append(train_dict[key])
            with open(str(output_data_path.joinpath(f'{client_id}.pkl')), 'wb') as handle:
                pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(str(base_data_path.joinpath('dev.pkl')), "rb") as f: dev_data = pickle.load(f)
        with open(str(base_data_path.joinpath('test.pkl')), "rb") as f: test_data = pickle.load(f)

        with open(str(output_data_path.joinpath(f'dev.pkl')), 'wb') as handle:
            pickle.dump(dev_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(str(output_data_path.joinpath(f'test.pkl')), 'wb') as handle:
            pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
