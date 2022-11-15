# Author: Tiantian Feng, USC SAIL lab, tiantiaf@usc.edu
import sys, os
import pickle
import re, pdb
import argparse
import torchaudio
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import KFold


sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
from partition_manager import partition_manager


def data_partition(args: dict):
    
    # define partition manager
    pm = partition_manager(args)
    
    # fetch all labels
    pm.fetch_label_dict()
    
    # save the partition
    output_data_path = Path(args.output_partition_path).joinpath(args.dataset)
    Path.mkdir(output_data_path, parents=True, exist_ok=True)
        
    # data root folder
    if Path.exists(output_data_path.joinpath(f'partition.pkl')) == False:
        client_id_list = [user.parts[-1].split('.')[0] for user in Path(args.raw_data_dir).joinpath('ExtraSensory/uuid_label').iterdir() if '.gz' in user.parts[-1]]
        client_id_list.sort()
        
        client_data_dict = dict()
        for client_id in client_id_list:
            # read data names
            client_data_dict[client_id] = list()
            label_df = pd.read_csv(Path(args.raw_data_dir).joinpath('ExtraSensory/uuid_label', f'{client_id}.features_labels.csv.gz'), index_col=0)[pm.label_dict]
            
            # iterate rows
            for index in tqdm(list(label_df.index), ncols=100, miniters=100):
                row_df = label_df.loc[index, :]
                if np.nansum(row_df) == 1:
                    acc_file_path = Path(args.raw_data_dir).joinpath('ExtraSensory/raw_acc', client_id, str(index)+'.m_raw_acc.dat')
                    gyro_file_path = Path(args.raw_data_dir).joinpath('ExtraSensory/proc_gyro', client_id, str(index)+'.m_proc_gyro.dat')
                    if Path.exists(acc_file_path) and Path.exists(gyro_file_path):
                        acc_data = np.genfromtxt(str(acc_file_path), dtype=float, delimiter=' ')[:, 1:]
                        if len(acc_data) != 800 and acc_data.shape[1] != 3: continue
                        gyro_data = np.genfromtxt(str(gyro_file_path), dtype=float, delimiter=' ')[:, 1:]
                        if len(gyro_data) != 800 and gyro_data.shape[1] != 3: continue
                        if len(acc_data) != len(gyro_data): continue
                        
                        label = pm.label_dict[row_df.index[row_df.argmax()]]
                        key = f'{client_id}/{str(index)}'
                        client_data_dict[client_id].append([key, str(acc_file_path), label])
        # pdb.set_trace()
        with open(str(output_data_path.joinpath(f'partition.pkl')), 'wb') as handle:
            pickle.dump(client_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # load processed partitions, for cv saving
    with open(str(output_data_path.joinpath(f'partition.pkl')), "rb") as f: 
        client_data_dict = pickle.load(f)
    
    for fold_idx in range(5):
        # read ids
        cv_path = Path(args.raw_data_dir).joinpath('ExtraSensory/cv_5_folds')
        with open(str(cv_path.joinpath(f'fold_{fold_idx}_train_android_uuids.txt'))) as f: train_android_ids = f.readlines()
        with open(str(cv_path.joinpath(f'fold_{fold_idx}_train_iphone_uuids.txt'))) as f: train_iphone_ids = f.readlines()
        with open(str(cv_path.joinpath(f'fold_{fold_idx}_test_android_uuids.txt'))) as f: test_android_ids = f.readlines()
        with open(str(cv_path.joinpath(f'fold_{fold_idx}_test_iphone_uuids.txt'))) as f: test_iphone_ids = f.readlines()
        
        # get all ids
        train_ids, test_ids = list(), list()
        for client_id in train_android_ids: train_ids.append(client_id[:-1])
        for client_id in train_iphone_ids: train_ids.append(client_id[:-1])
        for client_id in test_android_ids: test_ids.append(client_id[:-1])
        for client_id in test_iphone_ids: test_ids.append(client_id[:-1])

        # iterate over ids
        partition_dict = dict()
        partition_dict['dev'] = list()
        partition_dict['test'] = list()
        for client_id in train_ids:
            # read all keys
            train_val_arr = np.arange(len(client_data_dict[client_id]))
            if len(train_val_arr) == 0: continue
            
            # split train and dev from a client for later
            train_arr, dev_arr = pm.split_train_dev(train_val_arr)
            partition_dict[client_id] = list()
            for idx in train_arr: partition_dict[client_id].append(client_data_dict[client_id][idx])
            for idx in dev_arr: partition_dict['dev'].append(client_data_dict[client_id][idx])
        
        for client_id in test_ids:
            # read all keys
            arr = np.arange(len(client_data_dict[client_id]))
            if len(arr) == 0: continue
            arr.sort()
            for idx in arr: partition_dict['test'].append(client_data_dict[client_id][idx])

        # save the partition
        output_data_path = Path(args.output_partition_path).joinpath(args.dataset, f'fold{fold_idx+1}')
        Path.mkdir(output_data_path, parents=True, exist_ok=True)
        with open(output_data_path.joinpath(f'partition.pkl'), 'wb') as handle:
            pickle.dump(partition_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default="/media/data/public-data/HAR",
        help="Raw data path of extrasensory data set",
    )
    
    parser.add_argument(
        "--output_partition_path",
        type=str,
        default="/media/data/projects/speech-privacy/fed-multimodal/partition",
        help="Output path of speech_commands data set",
    )

    parser.add_argument(
        "--setup",
        type=str,
        default="federated",
        help="setup of the experiment: centralized/federated",
    )
    
    parser.add_argument(
        "--dataset", 
        type=str,
        default="extrasensory",
        help="data set name",
    )
    args = parser.parse_args()
    data_partition(args)
    
    
    