import os
import pdb
import glob
import random
import collections
import numpy as np
import os.path as osp
from tqdm import tqdm
from pathlib import Path


class partition_manager():
    def __init__(self, args: dict):
        self.args = args
    
    def fetch_filelist(self):
        # fetch file list
        if self.args.dataset == "ucf101":
            # read all files
            self.file_list = glob.glob(
                self.args.raw_data_dir + '/ucf101/audios/*/*.wav'
            )
            # raise error when no files found
            if len(self.file_list) == 0: 
                raise FileNotFoundError('No files exists at the location specified')
            self.file_list.sort()
        elif self.args.dataset in ["mit10", "mit51", "mit101"]:
            # read trani/test files
            self.train_file_list = glob.glob(
                self.args.raw_data_dir + '/mit/audios/training/*/*.wav'
            )
            self.test_file_list = glob.glob(
                self.args.raw_data_dir + '/mit/audios/validation/*/*.wav'
            )
            # raise error when no files found
            if len(self.train_file_list) == 0: 
                raise FileNotFoundError('No files exists at the location specified')
            self.train_file_list.sort()
            self.test_file_list.sort()
        elif self.args.dataset == "crema_d":
            self.file_list = [x for x in Path(self.args.raw_data_dir).joinpath(self.args.dataset, 'CREMA-D', "AudioWAV").iterdir() if '.wav' in x.parts[-1]]
            self.file_list.sort()
            
    def fetch_label_dict(self):
        # fetch unique labels
        if self.args.dataset == "ucf101":
            unique_labels = [path.split('/')[-2] for path in self.file_list]
            unique_labels = list(set(unique_labels))
            unique_labels.sort()
            self.label_dict = {k: i for i, k in enumerate(unique_labels)}
        elif self.args.dataset == "mit51":
            # we retrieve the most 51 frequently occured instances in MIT dataset
            full_labels = [path.split('/')[-2] for path in self.train_file_list]
            label_frequency = collections.Counter(full_labels)
            top51_label_frequency = label_frequency.most_common(51)
            unique_labels = list(set([label for label, freq in top51_label_frequency]))
            unique_labels.sort()
            self.label_dict = {k: i for i, k in enumerate(unique_labels)}
        elif self.args.dataset == "mit10":
            # we retrieve the most 10 frequently occured instances in MIT dataset
            full_labels = [path.split('/')[-2] for path in self.train_file_list]
            label_frequency = collections.Counter(full_labels)
            top10_label_frequency = label_frequency.most_common(10)
            unique_labels = list(set([label for label, freq in top10_label_frequency]))
            unique_labels.sort()
            self.label_dict = {k: i for i, k in enumerate(unique_labels)}
        elif self.args.dataset == "mit101":
            # we retrieve the most 101 frequently occured instances in MIT dataset
            full_labels = [path.split('/')[-2] for path in self.train_file_list]
            label_frequency = collections.Counter(full_labels)
            top101_label_frequency = label_frequency.most_common(101)
            unique_labels = list(set([label for label, freq in top101_label_frequency]))
            unique_labels.sort()
            self.label_dict = {k: i for i, k in enumerate(unique_labels)}
        elif self.args.dataset == "meld":
            self.label_dict = {
                'neutral': 0, 
                'sadness': 1, 
                'joy': 2, 
                'anger': 3
            }
        elif self.args.dataset == 'ku-har':
            self.label_dict  = {
                '0.Stand': 0, 
                '1.Sit': 1, 
                '11.Walk': 2, 
                '5.Lay': 3, 
                '8.Jump': 4, 
                '14.Run': 5, 
                '15.Stair-up': 6, 
                '16.Stair-down': 7
            }
        elif self.args.dataset in ['extrasensory', 'extrasensory_watch']:
            self.label_dict = {
                'label:SITTING': 0, 
                'label:FIX_walking': 1,
                'label:BICYCLING': 2,
                'label:OR_standing': 3,
                'label:PHONE_ON_TABLE': 6,
                'label:PHONE_IN_BAG': 7,
                'label:PHONE_IN_HAND': 8,
                'label:PHONE_IN_POCKET': 9
                # 'label:LYING_DOWN': 1,
                # 'label:LYING_DOWN': 1,
            }
            
            self.har_label_dict = {
                'label:SITTING': 0, 
                'label:FIX_walking': 1,
                'label:BICYCLING': 2,
                'label:OR_standing': 3
            }
        elif self.args.dataset in ['ptb-xl']:
            self.label_dict = {
                'NORM': 0, 
                'MI': 1, 
                'STTC': 2, 
                'CD': 3,
                'HYP': 4
            }
        elif self.args.dataset == "crema_d":
            self.label_dict = {
                "A": 0, 
                "S": 1, 
                "H": 2, 
                "N": 3
            }
        elif self.args.dataset == "crisis-mmd":
            self.label_dict = {
                'not_humanitarian':                         0, 
                'infrastructure_and_utility_damage':        1,
                'vehicle_damage':                           2, 
                'rescue_volunteering_or_donation_effort':   3,
                'other_relevant_information':               4, 
                'affected_individuals':                     5,
                'injured_or_dead_people':                   6, 
                'missing_or_found_people':                  7
            }

        elif self.args.dataset == 'uci-har':
            self.label_dict = {k: i for i, k in enumerate(np.arange(6))}
        elif self.args.dataset == 'hateful_memes':
            self.label_dict = {k: i for i, k in enumerate(np.arange(6))}    
        
    def split_train_dev(
        self, 
        train_val_file_id: list,
        seed: int=8
    ) -> (list, list):
        # shuffle train idx, and select 20% for dev
        train_arr = np.arange(len(train_val_file_id))
        np.random.seed(seed)
        np.random.shuffle(train_arr)
        val_len = int(len(train_arr)/5)
        # read the keys
        train_file_id = [train_val_file_id[idx] for idx in train_arr[val_len:]]
        val_file_id = [train_val_file_id[idx] for idx in train_arr[:val_len]]
        return train_file_id, val_file_id
    
    def direchlet_partition(
        self, 
        file_label_list: list,
        seed: int=8,
        min_sample_size: int=5
    ) -> (list):
        
        # cut the data using dirichlet
        min_size = 0
        K, N = len(np.unique(file_label_list)), len(file_label_list)
        # seed
        np.random.seed(seed)
        while min_size < min_sample_size:
            file_idx_clients = [[] for _ in range(self.args.num_clients)]
            for k in range(K):
                idx_k = np.where(np.array(file_label_list) == k)[0]
                np.random.shuffle(idx_k)
                # if self.args.dataset == "hateful_memes" and k == 0:
                #    proportions = np.random.dirichlet(np.repeat(1.0, self.args.num_clients))
                # else:
                proportions = np.random.dirichlet(np.repeat(self.args.alpha, self.args.num_clients))
                # Balance
                proportions = np.array([p*(len(idx_j)<N/self.args.num_clients) for p, idx_j in zip(proportions, file_idx_clients)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                file_idx_clients = [idx_j + idx.tolist() for idx_j,idx in zip(file_idx_clients,np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in file_idx_clients])
        return file_idx_clients
