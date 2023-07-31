# Author: Tiantian Feng
# USC SAIL lab, tiantiaf@usc.edu
import pdb
import pickle
import os, sys
import logging
import argparse

from tqdm import tqdm
from pathlib import Path

from fed_multimodal.features.feature_processing.feature_manager import FeatureManager

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
        default='mobilebert',
        type=str, 
        help='output feature name'
    )

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
    fm = FeatureManager(args)
    
    # fetch all files for processing
    partition_dict = fm.fetch_partition()
    logging.info(f'Reading text from folder: {args.raw_data_dir}')
    logging.info(f'Total number of clients found: {len(partition_dict.keys())}')
    
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


