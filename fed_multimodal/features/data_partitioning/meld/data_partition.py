import os
import json
import argparse
import pandas as pd
import argparse, pickle, pdb, re

from tqdm import tqdm
from pathlib import Path

from fed_multimodal.features.data_partitioning.partition_manager import PartitionManager


def data_partition(
    args: dict, 
    split: str 
) -> (dict):
    """
    Gets wav data dict and the number of unique classes for task.
    :param args: argument input
    :param split: train/dev/test
    return: data_dict => 
            {speaker_id: [[speaker_id, wav_file_name, wav_file_path, label, text_data], ...]}
    """
    if split == 'train':
        label_path = f'{args.raw_data_dir}/meld/MELD.Raw/train_sent_emo.csv'
        data_path = f'{args.raw_data_dir}/meld/MELD.Raw/train_splits'
    elif split == 'test':
        label_path = f'{args.raw_data_dir}/meld/MELD.Raw/test_sent_emo.csv'
        data_path = f'{args.raw_data_dir}/meld/MELD.Raw/output_repeated_splits_test'
    elif split == 'dev':
        label_path = f'{args.raw_data_dir}/meld/MELD.Raw/dev_sent_emo.csv'
        data_path = f'{args.raw_data_dir}/meld/MELD.Raw/dev_splits_complete'
        
    df_label = pd.read_csv(label_path)
    err = []
    for i, df_row in tqdm(df_label.iterrows()):
        if not Path(f"{data_path}/waves/dia{df_row.Dialogue_ID}_utt{df_row.Utterance_ID}.wav").is_file():
            err.append(i)
    print(f'Missing/Corrupt files for indices: {err}')
    df_label_cleaned = df_label.drop(err)
    df_label_cleaned['Label'] = df_label_cleaned.Emotion.apply(lambda x: pm.label_dict[x] if x in pm.label_dict else 7)
    df_label_cleaned = df_label_cleaned.loc[df_label_cleaned['Label']<7]
        
    df_label_cleaned['Path'] = df_label_cleaned.apply(lambda row: f"{data_path}/waves/dia{row.Dialogue_ID}_utt{row.Utterance_ID}.wav", axis=1)
    df_label_cleaned['Filename'] = df_label_cleaned.apply(lambda row: f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}", axis=1)
    df_label_cleaned['SeasondSpeaker'] = df_label_cleaned.apply(lambda row: f"season{row.Season}_utt{row.Speaker}", axis=1)
    
    df_label_reduced = df_label_cleaned[['SeasondSpeaker', 'Filename', 'Path', 'Label', 'Utterance']]
    groups = df_label_reduced.groupby('SeasondSpeaker')
    data_dict = {speaker: group[['Filename', 'Path', 'Label', 'Utterance']].values.tolist()
    for _, (speaker, group) in enumerate(groups) if len(group[['Filename', 'Path', 'Label', 'SeasondSpeaker', 'Utterance']]) > 10}
    
    for filter_speaker in ["All", "Man", "Policeman", "Tag", "Woman"]:
        if filter_speaker in data_dict: data_dict.pop(filter_speaker)

    return data_dict
        
    
if __name__ == '__main__':

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

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=path_conf['data_dir'],
        help="Raw data path of speech_commands data set",
    )
    
    parser.add_argument(
        "--output_partition_path",
        type=str,
        default=path_conf['output_dir'],
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
        default="meld",
        help="dataset name",
    )
    args = parser.parse_args()

    # save the partition
    output_data_path = Path(args.output_partition_path).joinpath('partition', args.dataset)
    Path.mkdir(output_data_path, parents=True, exist_ok=True)

    # define partition manager
    pm = PartitionManager(args)
    
    # fetch all labels
    pm.fetch_label_dict()
    partition_dict = dict()

    # partition
    train_dict = data_partition(args, split='train')
    dev_dict = data_partition(args, split='dev')
    test_dict = data_partition(args, split='test')

    # train data
    client_keys = list(train_dict.keys())
    client_keys.sort()

    for client_idx in range(len(train_dict)): 
        partition_dict[client_idx] = train_dict[client_keys[client_idx]]
    
    # dev data
    client_keys = list(dev_dict.keys())
    client_keys.sort()

    partition_dict['dev'] = list()
    for client_idx in range(len(dev_dict)):
        client_key = client_keys[client_idx]
        for idx in range(len(dev_dict[client_key])): 
            partition_dict['dev'].append(dev_dict[client_key][idx])
    
    # dev data
    client_keys = list(test_dict.keys())
    client_keys.sort()

    partition_dict['test'] = list()
    for client_idx in range(len(test_dict)):
        client_key = client_keys[client_idx]
        for idx in range(len(test_dict[client_key])):
            partition_dict['test'].append(test_dict[client_key][idx])

    # dump the dictionary
    with open(output_data_path.joinpath(f'partition.json'), "w") as handle:
        json.dump(
            partition_dict, 
            handle
        )
