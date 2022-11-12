import os
import random
import glob
import pdb
import numpy as np
import os.path as osp
from tqdm import tqdm
from pathlib import Path


class partition_manager():
    def __init__(self, args: dict):
        self.args = args
    
    def fetch_filelist(self):
        if self.args.dataset == "ucf101":
            self.file_list = glob.glob(self.args.raw_data_dir + '/audios/*/*.wav')
            self.file_list.sort()
            
    def fetch_label_dict(self):
        if self.args.dataset == "ucf101":
            unique_labels = [path.split('/')[-2] for path in self.file_list]
            unique_labels = list(set(unique_labels))
            unique_labels.sort()
            self.label_dict = {k: i for i, k in enumerate(unique_labels)}
        
    def split_train_dev(self, train_val_file_id: list) -> (list, list):
        train_arr = np.arange(len(train_val_file_id))
        np.random.seed(8)
        np.random.shuffle(train_arr)
        val_len = int(len(train_arr)/5)
        
        train_file_id = [train_val_file_id[idx] for idx in train_arr[val_len:]]
        val_file_id = [train_val_file_id[idx] for idx in train_arr[:val_len]]
        
        return train_file_id, val_file_id
    
    def direchlet_partition(self, file_label_list: list) -> (list):
        # cut the data using dirichlet
        min_size = 0
        K, N = len(np.unique(file_label_list)), len(file_label_list)
        # at least we train 1 full batch
        min_sample_size = 5
        np.random.seed(8)
        while min_size < min_sample_size:
            file_idx_clients = [[] for _ in range(self.args.num_clients)]
            for k in range(K):
                idx_k = np.where(np.array(file_label_list) == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.args.alpha, self.args.num_clients))
                # Balance
                proportions = np.array([p*(len(idx_j)<N/self.args.num_clients) for p, idx_j in zip(proportions, file_idx_clients)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                file_idx_clients = [idx_j + idx.tolist() for idx_j,idx in zip(file_idx_clients,np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in file_idx_clients])
        return file_idx_clients

