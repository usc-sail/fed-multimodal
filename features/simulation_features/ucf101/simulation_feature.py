# Author: Tiantian Feng, USC SAIL lab, tiantiaf@usc.edu
import argparse
import glob
import os, sys
import os.path as osp
import pdb
import torch
import random
import torchaudio
import numpy as np
import pickle
import opensmile
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
from simulation_manager import simulation_manager


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Simulation Features')
    parser.add_argument(
        '--raw_data_dir',
        default='/media/data/public-data/MMAction/ucf101', 
        type=str,
        help='source data directory')
    parser.add_argument(
        '--output_dir', 
        default='/media/data/projects/speech-privacy/fed-multimodal/',
        type=str, 
        help='output feature directory')
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="alpha in direchlet distribution",
    )
    
    parser.add_argument(
        "--missing_modality",
        type=bool, 
        default=False,
        help="missing modality simulation",
    )
    
    parser.add_argument(
        "--en_missing_modality",
        dest='missing_modality',
        action='store_true',
        help="enable missing modality simulation",
    )
    
    parser.add_argument(
        "--missing_modailty_rate",
        type=float, 
        default=0.5,
        help='missing rate for modality; 0.9 means 90%% missing'
    )
    
    parser.add_argument(
        "--missing_label",
        type=bool, 
        default=False,
        help="missing label simulation",
    )
    
    parser.add_argument(
        "--en_missing_label",
        dest='missing_label',
        action='store_true',
        help="enable missing label simulation",
    )
    
    parser.add_argument(
        "--missing_label_rate",
        type=float, 
        default=0.5,
        help='missing rate for modality; 0.9 means 90%% missing'
    )
    
    parser.add_argument(
        '--label_nosiy', 
        type=bool, 
        default=False,
        help='clean label or nosiy label')
    
    parser.add_argument(
        "--en_label_nosiy",
        dest='label_nosiy',
        action='store_true',
        help="enable label noise simulation",
    )

    parser.add_argument(
        '--label_nosiy_level', 
        type=float, 
        default=0.1,
        help='nosiy level for labels; 0.9 means 90% wrong')
    
    parser.add_argument("--dataset", default="ucf101")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # read args
    args = parse_args()
    alpha_str = str(args.alpha).replace('.', '')
    
    # initialize simulation manager
    sm = simulation_manager(args)
    
    # save for later uses
    for fold_idx in range(3):
        # define output path
        output_data_path = Path(args.output_dir).joinpath('simulation_feature', 
                                                          args.dataset, 
                                                          f'fold{fold_idx+1}')
        Path.mkdir(output_data_path, parents=True, exist_ok=True)
        partition_dict = sm.fetch_partition(fold_idx+1, alpha=args.alpha)
        
        for client_idx, client in enumerate(partition_dict):
            partition_dict[client] = sm.simulation(partition_dict[client], seed=client_idx)
            
        sm.get_simulation_setting(alpha=args.alpha)
        if len(sm.setting_str) != 0:
            with open(output_data_path.joinpath(f'{sm.setting_str}.pkl'), 'wb') as handle:
                pickle.dump(partition_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            