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
    if batch[0][0] is not None: max_a_len = max(map(lambda x: x[0].shape[0], batch))
    if batch[0][1] is not None: max_b_len = max(map(lambda x: x[1].shape[0], batch))

    # pad according to max_len
    x_a, x_b, ys = list(), list(), list()
    for idx in range(len(batch)):
        if batch[0][0] is not None:
            x_a.append(pad_tensor(batch[idx][0], pad=max_a_len))
        else:
            x_a.append(None)
        if batch[0][1] is not None:
            x_b.append(pad_tensor(batch[idx][1], pad=max_b_len))
        else:
            x_b.append(None)
        ys.append(batch[idx][2])
    
    # stack all
    if batch[0][0] is not None: x_a = torch.stack(x_a, dim=0)
    if batch[0][1] is not None: x_b = torch.stack(x_b, dim=0)
    ys = torch.stack(ys, dim=0)
    return x_a, x_b, ys


class MMDatasetGenerator(Dataset):
    def __init__(self, modalityA, modalityB, data_len, simulate_feat=None):
        self.modalityA = modalityA
        self.modalityB = modalityB
        self.simulate_feat = simulate_feat
        self.data_len = data_len

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        
        # read modality
        data_a = self.modalityA[item][-1]
        data_b = self.modalityB[item][-1]
        label = torch.tensor(self.modalityA[item][-2])
        
        # modality A
        if data_a is not None: 
            data_a = torch.tensor(data_a)

        # modality B
        if data_b is not None: 
            data_b = torch.tensor(data_b)
        return data_a, data_b, label


class dataload_manager():
    def __init__(self, args: dict):
        self.args = args
        # initialize paths
        self.get_audio_feat_path()
        self.get_video_feat_path()
        
    def get_audio_feat_path(self):
        self.audio_feat_path = Path(self.args.data_dir).joinpath('feature', 'audio', self.args.audio_feat, self.args.dataset)
        return Path(self.audio_feat_path)
    
    def get_video_feat_path(self):
        self.video_feat_path = Path(self.args.data_dir).joinpath('feature', 'video', self.args.video_feat, self.args.dataset)
        return Path(self.video_feat_path)
    
    def load_audio_feat(self, fold_idx=1, split_type='train'):
        if self.args.dataset == "ucf101":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.audio_feat_path.joinpath(f'alpha{alpha_str}', 
                                                      f'fold{fold_idx}', 
                                                      f'{split_type}.pkl')
        
        with open(str(data_path), "rb") as f: 
            data_dict = pickle.load(f)
            if split_type == 'train':
                self.train_audio = pickle.load(f)
            elif split_type == 'train':
                self.train_audio = pickle.load(f)
            elif split_type == 'train':
                self.train_audio = pickle.load(f)
        return data_dict
    
    def load_full_audio_feat(self, fold_idx: int=1):
        if self.args.dataset == "ucf101":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.audio_feat_path.joinpath(f'alpha{alpha_str}', f'fold{fold_idx}')
        
            with open(str(data_path.joinpath('train.pkl')), "rb") as f: 
                self.train_audio = pickle.load(f)
            with open(str(data_path.joinpath('dev.pkl')), "rb") as f: 
                self.dev_audio = pickle.load(f)
            with open(str(data_path.joinpath('test.pkl')), "rb") as f: 
                self.test_audio = pickle.load(f)
        elif self.args.dataset == "mit51":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.audio_feat_path.joinpath(f'alpha{alpha_str}')


    def load_video_feat(self, 
                        fold_idx: int=1, 
                        split_type: str='train') -> dict:
        if self.args.dataset == "ucf101":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.video_feat_path.joinpath(f'alpha{alpha_str}', 
                                                      f'fold{fold_idx}', 
                                                      f'{split_type}.pkl')

        with open(str(data_path), "rb") as f: 
            data_dict = pickle.load(f)
        return data_dict

    def load_full_video_feat(self, fold_idx: int=1):
        if self.args.dataset == "ucf101":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.video_feat_path.joinpath(f'alpha{alpha_str}', f'fold{fold_idx}')

            with open(str(data_path.joinpath('train.pkl')), "rb") as f: 
            self.train_video = pickle.load(f)
            with open(str(data_path.joinpath('dev.pkl')), "rb") as f: 
                self.dev_video = pickle.load(f)
            with open(str(data_path.joinpath('test.pkl')), "rb") as f: 
                self.test_video = pickle.load(f)

        elif self.args.dataset == "mit51":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.video_feat_path.joinpath(f'alpha{alpha_str}')
        
        

    def set_dataloader(self, client_id, shuffle=False):
        # read data
        if client_id == 'dev':
            data_a = self.dev_audio
            data_b = self.dev_video
        elif client_id == 'test':
            data_a = self.test_audio
            data_b = self.test_video
        else:
            data_a = self.train_audio[client_id]
            data_b = self.train_video[client_id]

        # modify data based on simulation
        if self.sim_data is not None:
            for idx in range(len(self.sim_data[client_id])):
                # read simulate feature
                sim_data = self.sim_data[client_id][idx][-1]
                # read modality A
                if read[0] == 1: data_a[idx][-1] = None
                # read modality B
                if read[1] == 1: data_b[idx][-1] = None
                # label noise
                data_a[idx][-2] = read[2]
        
        data_len = len(data_a)
        data_ab = MMDatasetGenerator(data_a, data_b, data_len)
        dataloader = DataLoader(data_ab, batch_size=int(self.args.batch_size), num_workers=0, shuffle=shuffle, collate_fn=collate_mm_fn_padd)
        return dataloader
    
    def load_sim_dict(self, fold_idx=1):
        """
        Load simulation dictionary.
        :param fold_idx: fold index
        :return: None
        """
        if self.setting_str == '': 
            self.sim_data = None
            return
        
        if self.args.dataset == "ucf101":
            data_path = Path(self.args.data_dir).joinpath('simulation_feature',
                                                          self.args.dataset, 
                                                          f'fold{fold_idx}', 
                                                          f'{self.setting_str}.pkl')
        elif self.args.dataset == "mit51":
            data_path = Path(self.args.data_dir).joinpath('simulation_feature',
                                                          self.args.dataset,
                                                          f'{self.setting_str}.pkl')

        with open(str(data_path), "rb") as f: 
            self.sim_data = pickle.load(f)
    
    def get_simulation_setting(self, alpha=None):
        self.setting_str = ''
        if self.args.missing_modality == True:
            self.setting_str += 'mm'+str(self.args.missing_modailty_rate).replace('.', '')
        if self.args.label_nosiy == True:
            if len(self.setting_str) != 0: self.setting_str += '_'
            self.setting_str += 'ln'+str(self.args.label_nosiy_level).replace('.', '')
        if self.args.missing_label == True:
            if len(self.setting_str) != 0: self.setting_str += '_'
            self.setting_str += 'ml'+str(self.args.missing_label_rate).replace('.', '')
        if len(self.setting_str) != 0:
            if alpha is not None:
                alpha_str = str(self.args.alpha).replace('.', '')
                self.setting_str += f'_alpha{alpha_str}'