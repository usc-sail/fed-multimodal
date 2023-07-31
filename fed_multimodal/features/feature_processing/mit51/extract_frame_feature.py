import os
import pdb
import pickle
import logging
import numpy as np
import argparse, sys
import os.path as osp

from tqdm import tqdm
from pathlib import Path

from fed_multimodal.features.feature_processing.feature_manager import FeatureManager


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

    parser = argparse.ArgumentParser(description='Extract frame features')
    parser.add_argument(
        '--raw_data_dir',
        default=path_conf["data_dir"], 
        type=str,
        help='source data directory'
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
    feature_manager = FeatureManager(args)

    # fetch all files for processing
    partition_dict = feature_manager.fetch_partition(alpha=args.alpha)
    base_partition_dict = feature_manager.fetch_partition(alpha=1.0)

    print('Reading videos from folder: ', args.raw_data_dir)
    print('Total number of videos found: ', len(partition_dict.keys()))
    
    # extract data
    base_data_path = Path(args.output_dir).joinpath('feature', 'video', args.feature_type, args.dataset, f'alpha10')
    base_client_file_paths = os.listdir(base_data_path)
    base_client_file_paths.sort()

    if len(base_client_file_paths) != len(base_partition_dict) and args.alpha == 1.0:
        for client in tqdm(list(base_partition_dict.keys())[809:810]):
            data_dict = base_partition_dict[client].copy()
            split = 'validation' if client == 'test' else 'training'
            if Path.exists(output_data_path.joinpath(f'{client}.pkl')) == True: continue
            for idx in range(len(base_partition_dict[client])):
                file_path = base_partition_dict[client][idx][1]
                video_id, _ = osp.splitext(osp.basename(file_path))
                label_str = osp.basename(osp.dirname(file_path))
                features = feature_manager.extract_frame_features(video_id, label_str, max_len=8, split=split)
                
                # data_dict[f'{label_str}/{video_id}'] = features
                data_dict[idx].append(features)
            # saving features
            # pdb.set_trace()
            with open(str(output_data_path.joinpath(f'{client}.pkl')), 'wb') as handle:
                pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save for alpha != 1.0
    if len(base_client_file_paths) == len(base_partition_dict) and args.alpha != 1.0:
        train_dict = dict()
        logging.info('Read alpha=1.0 data')
        for client_file_path in tqdm(base_client_file_paths[:-2]):
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
    

