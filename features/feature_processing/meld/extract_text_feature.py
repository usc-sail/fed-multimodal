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
    parser = argparse.ArgumentParser(description='Extract text features')
    parser.add_argument(
        '--raw_data_dir',
        default='/media/data/public-data/SER/meld/MELD.Raw', 
        type=str,
        help='source data directory')

    parser.add_argument(
        '--output_dir', 
        default='/media/data/projects/speech-privacy/fed-multimodal/',
        type=str, 
        help='output feature directory')

    parser.add_argument(
        '--feature_type', 
        default='bert',
        type=str, 
        help='output feature name')

    parser.add_argument("--run_extraction", default=True, action='store_true')
    parser.add_argument('--skip_extraction', dest='run_extraction', action='store_false')
    parser.add_argument("--dataset", default="meld")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # read args
    args = parse_args()
    output_data_path = Path(args.output_dir).joinpath('feature', 'text', args.feature_type, args.dataset)
    Path.mkdir(output_data_path, parents=True, exist_ok=True)
    
    # initialize feature processer
    fm = feature_manager(args)
    
    # fetch all files for processing
    partition_dict = fm.fetch_partition()
    print('Reading videos from folder: ', args.raw_data_dir)
    print('Total number of videos found: ', len(partition_dict.keys()))
    
    # extract data
    for client in partition_dict:
        data_dict = partition_dict[client].copy()
        if Path.exists(output_data_path.joinpath(f'{client}.pkl')) == True: continue
        for idx in tqdm(range(len(partition_dict[client]))):
            text_str = partition_dict[client][idx][-1]
            features = fm.extract_text_feature(input_str=text_str)
            data_dict[idx][-1] = features
        # saving features
        with open(str(output_data_path.joinpath(f'{client}.pkl')), 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


