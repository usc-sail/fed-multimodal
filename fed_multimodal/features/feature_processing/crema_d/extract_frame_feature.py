# Author: Tiantian Feng
# USC SAIL lab, tiantiaf@usc.edu
import pdb
import pickle
import os, sys
import logging
import warnings
import argparse

from tqdm import tqdm
from pathlib import Path
from moviepy.editor import *

from fed_multimodal.features.feature_processing.feature_manager import FeatureManager

warnings.filterwarnings('ignore')

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

    parser = argparse.ArgumentParser(description='Extract audio features')
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
        '--feature_type', 
        default='mobilenet_v2',
        type=str, 
        help='output feature name'
    )

    parser.add_argument(
        "--dataset", 
        default="crema_d"
    )
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # read args
    args = parse_args()
    # read the base case
    base_data_path = Path(args.output_dir).joinpath(
        'feature', 
        'video', 
        args.feature_type, 
        args.dataset, 
        'fold1'
    )
    Path.mkdir(
        base_data_path, 
        parents=True, 
        exist_ok=True
    )
    base_file_paths = os.listdir(base_data_path)
    base_file_paths.sort()
    
    # initialize feature processer
    fm = FeatureManager(args)
    
    # read base_partition keys
    base_partition_dict = fm.fetch_partition(
        fold_idx=1,
        file_ext="json"
    )
        
    # extract based feature, fold1 is the base case
    if len(base_file_paths) != len(base_partition_dict):
    
        logging.info(f'Reading audios from folder: {args.raw_data_dir}')
        logging.info(f'Total number of audios found: {len(base_partition_dict.keys())}')
        
        # extract data
        for client_id in base_partition_dict:
            if Path.exists(base_data_path.joinpath(f'{client_id}.pkl')) == True: continue
            data_dict = base_partition_dict[client_id].copy()
            # normal client case each speaker is a client
            logging.info(f'process data for {client_id}')
            for idx in tqdm(range(len(base_partition_dict[client_id]))):
                # convert audio path to video path
                file_path = base_partition_dict[client_id][idx][1]
                file_path = file_path.replace("AudioWAV", "VideoFlash")
                file_path = file_path.replace(".wav", ".flv")
                
                # read video data
                features = fm.extract_frame_features_ser(
                    video_path=file_path
                )
                # the last one was speaker id: str, replace with feature instead
                data_dict[idx][-1] = features

            # saving features
            with open(str(base_data_path.joinpath(f'{client_id}.pkl')), 'wb') as handle:
                pickle.dump(
                    data_dict, 
                    handle, 
                    protocol=pickle.HIGHEST_PROTOCOL
                )

    # iterate over folds
    base_file_paths = os.listdir(base_data_path)
    base_file_paths.sort()
    
    for fold_idx in range(2, 6):
        # read partition keys
        partition_dict = fm.fetch_partition(
            fold_idx=fold_idx,
            file_ext="json"
        )
        logging.info(f'Reading videos from folder: {args.raw_data_dir}')
        logging.info(f'Total number of videos found: {len(partition_dict.keys())}')
        
        # read output path
        output_data_path = Path(args.output_dir).joinpath(
            'feature', 
            'video', 
            args.feature_type, 
            args.dataset,
            f'fold{fold_idx}'
        )
        # make output path
        Path.mkdir(
            output_data_path, 
            parents=True, 
            exist_ok=True
        )
        
        if len(base_file_paths) == len(base_partition_dict):
            data_dict = dict()
            logging.info(f'Read base case data')
            # read data key: [key, file_path, label, feature]
            for file_path in tqdm(base_file_paths):
                with open(str(base_data_path.joinpath(file_path)), "rb") as f: 
                    client_data = pickle.load(f)
                for idx in range(len(client_data)):
                    key = client_data[idx][0]
                    data_dict[key] = client_data[idx]
                    
            # reading data
            logging.info(f'Read fold idx={fold_idx} data')
            client_ids = list(partition_dict.keys())
            for client_id in tqdm(client_ids):
                save_data = list()
                for idx in range(len(partition_dict[client_id])):
                    key = partition_dict[client_id][idx][0]
                    save_data.append(data_dict[key])
                with open(str(output_data_path.joinpath(f'{client_id}.pkl')), 'wb') as handle:
                    pickle.dump(
                        save_data, 
                        handle, 
                        protocol=pickle.HIGHEST_PROTOCOL
                    )
        
        

