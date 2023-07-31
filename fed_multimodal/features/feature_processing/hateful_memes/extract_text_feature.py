import os
import pdb
import pickle
import logging
import argparse, sys

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

    parser = argparse.ArgumentParser(description='Extract text features')
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
        default='mobilebert',
        type=str, 
        help='output feature name'
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="alpha in direchlet distribution",
    )
    parser.add_argument("--dataset", default="hateful_memes")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    # read args
    args = parse_args()
    alpha_str = str(args.alpha).replace('.', '')
    output_data_path = Path(args.output_dir).joinpath(
        'feature', 
        'text', 
        args.feature_type, 
        args.dataset, 
        f'alpha{alpha_str}'
    )
    Path.mkdir(output_data_path, parents=True, exist_ok=True)
    
    # initialize feature processer
    fm = FeatureManager(args)

    # fetch all files for processing
    partition_dict = fm.fetch_partition(alpha=args.alpha)
    print(f'Total number of clients found: {len(partition_dict.keys())}')
    
    # extract data, read base case first, alpha=1.0
    # If the base folder is empty, we extract for the base case
    base_data_path = Path(args.output_dir).joinpath(
        'feature', 
        'text', 
        args.feature_type, 
        args.dataset, 
        f'alpha50'
    )
    client_file_paths = os.listdir(base_data_path)
    client_file_paths.sort()
    
    # extract based feature
    if len(client_file_paths) != len(partition_dict) and args.alpha == 5.0:
        # iterate over client, including keys = dev/test
        for client in tqdm(list(partition_dict.keys())):
            data_dict = partition_dict[client].copy()
            if Path.exists(output_data_path.joinpath(f'{client}.pkl')) == True: continue
            # extract feature
            for idx in range(len(partition_dict[client])):
                text_str = partition_dict[client][idx][-1]
                features = fm.extract_text_feature(
                    input_str=text_str
                )
                data_dict[idx][-1] = features
            # saving features
            with open(str(output_data_path.joinpath(f'{client}.pkl')), 'wb') as handle:
                pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # base feature all extracted, and we want to explore other alpha cases
    if len(client_file_paths) == len(partition_dict):
        train_dict = dict()
        logging.info('Read alpha=5.0 data')
        for client_file_path in tqdm(client_file_paths[:-2]):
            with open(str(base_data_path.joinpath(client_file_path)), "rb") as f: 
                client_data = pickle.load(f)
            for idx in range(len(client_data)):
                key = client_data[idx][0]
                train_dict[key] = client_data[idx]
        # reading data
        logging.info(f'Save alpha={args.alpha} data')
        client_ids = [client_id for client_id in list(partition_dict.keys()) if client_id not in ['dev', 'test']]
        for client_id in tqdm(client_ids):
            save_data = list()
            for idx in range(len(partition_dict[client_id])):
                key = partition_dict[client_id][idx][0]
                save_data.append(train_dict[key])
            with open(str(output_data_path.joinpath(f'{client_id}.pkl')), 'wb') as handle:
                pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # dev and test should be the same, as fix the seed when spliting the data
        with open(str(base_data_path.joinpath('dev.pkl')), "rb") as f: dev_data = pickle.load(f)
        with open(str(base_data_path.joinpath('test.pkl')), "rb") as f: test_data = pickle.load(f)
        # saving data
        with open(str(output_data_path.joinpath(f'dev.pkl')), 'wb') as handle:
            pickle.dump(dev_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(str(output_data_path.joinpath(f'test.pkl')), 'wb') as handle:
            pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

