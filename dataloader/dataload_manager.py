import torch
import pickle
import glob
import random
import pdb, os
import torchaudio
import numpy as np
import os.path as osp

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

def pad_tensor(vec, pad):
    pad_size = list(vec.shape)
    pad_size[0] = pad - vec.size(0)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=0)

def collate_mm_fn_padd(batch):
    # find longest sequence
    max_a_len = max(map(lambda x: x[0].shape[0], batch))
    max_b_len = max(map(lambda x: x[1].shape[0], batch))

    # pad according to max_len
    x_a, x_b, ys = list(), list(), list()
    for idx in range(len(batch)):
        x_a.append(pad_tensor(batch[idx][0], pad=max_a_len))
        x_b.append(pad_tensor(batch[idx][1], pad=max_b_len))
        ys.append(batch[idx][2])
    # stack all
    x_a, x_b, ys = torch.stack(x_a, dim=0), torch.stack(x_b, dim=0), torch.stack(ys, dim=0)
    return x_a, x_b, ys


class MMDatasetGenerator(Dataset):
    def __init__(self, modalityA, modalityB):
        self.modalityA = modalityA
        self.modalityB = modalityB

    def __len__(self):
        return len(self.modalityA)

    def __getitem__(self, item):
        return torch.tensor(self.modalityA[item][-1]), torch.tensor(self.modalityB[item][-1]), torch.tensor(self.modalityA[item][-2])


class dataload_manager():
    def __init__(self, args: dict):
        self.args = args
        # initialize paths
        self.get_audio_feat_path()
        self.get_video_feat_path()
        
    def get_audio_feat_path(self):
        self.audio_feat_path = Path(self.args.data_dir).joinpath('feature', 'audio', self.args.audio_feat, self.args.dataset)
        return Path(self.args.data_dir).joinpath('feature', 'audio', self.args.audio_feat, self.args.dataset)
    
    def get_video_feat_path(self):
        self.video_feat_path = Path(self.args.data_dir).joinpath('feature', 'video', self.args.video_feat, self.args.dataset)
        return Path(self.args.data_dir).joinpath('feature', 'video', self.args.video_feat, self.args.dataset)
    
    def load_audio_feat(self, fold_idx=1, split_type='train'):
        if self.args.dataset == "ucf101":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.audio_feat_path.joinpath(f'fold{fold_idx}', f'{split_type}_alpha{alpha_str}.pkl')

        with open(str(data_path), "rb") as f: 
            data_dict = pickle.load(f)
        return data_dict

    def load_video_feat(self, fold_idx=1, split_type='train'):
        if self.args.dataset == "ucf101":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.video_feat_path.joinpath(f'fold{fold_idx}', f'{split_type}_alpha{alpha_str}.pkl')

        with open(str(data_path), "rb") as f: 
            data_dict = pickle.load(f)
        return data_dict

    def set_dataloader(self, data_a, data_b, shuffle=False):
        data_ab = MMDatasetGenerator(data_a, data_b)
        dataloader = DataLoader(data_ab, batch_size=int(self.args.batch_size), num_workers=0, shuffle=shuffle, collate_fn=collate_mm_fn_padd)
        return dataloader