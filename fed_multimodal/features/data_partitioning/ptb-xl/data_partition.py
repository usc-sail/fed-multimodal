# Author: Tiantian Feng 
# USC SAIL lab, tiantiaf@usc.edu
import json
import sys, os
import re, pdb
import argparse
import wfdb, ast
import numpy as np
import pandas as pd
from pathlib import Path
from fed_multimodal.features.data_partitioning.partition_manager import PartitionManager

# Define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)
        

def data_partition(args: dict):
    
    # define partition manager
    pm = PartitionManager(args)
    
    # fetch all labels
    pm.fetch_label_dict()
    
    # save the partition
    output_data_path = Path(args.output_partition_path).joinpath(
        'partition',
        args.dataset
    )
    Path.mkdir(output_data_path, parents=True, exist_ok=True)
        
    # data root folder
    data_path = Path(args.raw_data_dir).joinpath(
        args.dataset,
        'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
    )
    
    patient_df = pd.read_csv(
        data_path.joinpath('ptbxl_database.csv'), 
        index_col=0
    )
    
    # Apple code
    patient_df = patient_df.dropna(subset=['site'])
    patient_df.scp_codes = patient_df.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in mapping_df.index and y_dic[key] == 100:
                tmp.append(mapping_df.loc[key].diagnostic_class)
        return list(set(tmp))
    # Read mapping
    mapping_df = pd.read_csv(
        data_path.joinpath('scp_statements.csv'), 
        index_col=0
    )
    mapping_df = mapping_df[mapping_df.diagnostic == 1]
    # Apply diagnostic superclass
    patient_df['diagnostic_superclass'] = patient_df.scp_codes.apply(aggregate_diagnostic)
    
    # the suggest split from the data repo:
    # Records in fold 9 and 10 underwent at least one human evaluation and are therefore of a particularly high label quality. 
    # We therefore propose to use folds 1-8 as training set, fold 9 as validation set and fold 10 as test set.
    train_df = patient_df.loc[patient_df['strat_fold'] <= 8]
    dev_df = patient_df.loc[patient_df['strat_fold'] == 9]
    test_df = patient_df.loc[patient_df['strat_fold'] == 10]
    
    # site as a data dict
    partition_dict = dict()
    partition_dict['dev'] = list()
    partition_dict['test'] = list()
    unique_sites = set(train_df['site'])
    
    # iterate over sites
    for site_id in unique_sites:
        # select site information
        site_df = train_df.loc[train_df.site == site_id]
        if len(site_df) < 10: continue
        logging.info(f'Processing data for {site_id}: data number: {len(site_df)}')
        
        # initialize site data
        partition_dict[site_id] = list()
        for idx in range(len(site_df)):
            # read file name
            file_name = site_df.filename_lr.values[idx]
            # labels
            raw_labels = site_df.diagnostic_superclass.values[idx]
            label = list(np.zeros(len(pm.label_dict)))
            for raw_label in raw_labels: label[pm.label_dict[raw_label]] = 1
            partition_dict[site_id].append([f'{site_id}/{file_name}', file_name, label])
    
    # dev condition
    for idx in range(len(dev_df)):
        # read file name
        file_name = dev_df.filename_lr.values[idx]
        # labels
        raw_labels = dev_df.diagnostic_superclass.values[idx]
        label = list(np.zeros(len(pm.label_dict)))
        for raw_label in raw_labels: label[pm.label_dict[raw_label]] = 1
        partition_dict['dev'].append([f'dev/{file_name}', file_name, label])
    
    # test condition
    for idx in range(len(test_df)):
        # read file name
        file_name = test_df.filename_lr.values[idx]
        # labels
        raw_labels = test_df.diagnostic_superclass.values[idx]
        label = list(np.zeros(len(pm.label_dict)))
        for raw_label in raw_labels: label[pm.label_dict[raw_label]] = 1
        partition_dict['test'].append([f'test/{file_name}', file_name, label])
        
    jsonString = json.dumps(partition_dict, indent=4)
    jsonFile = open(str(output_data_path.joinpath(f'partition.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()


if __name__ == "__main__":
    
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
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=path_conf["data_dir"],
        help="Raw data path of extrasensory data set",
    )
    
    parser.add_argument(
        "--output_partition_path",
        type=str,
        default=path_conf["output_dir"],
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
        default="ptb-xl",
        help="data set name",
    )
    args = parser.parse_args()
    data_partition(args)
    
    
    