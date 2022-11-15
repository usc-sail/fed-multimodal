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


class DataloadManager():
    def __init__(self, args: dict):
        self.args = args
        # Initialize video feature paths
        if self.args.dataset in ['ucf101', 'mit51', 'mit101']:
            self.get_video_feat_path()
        # Initialize audio feature paths
        if self.args.dataset in ['ucf101', 'mit51', 'mit101', 'meld']:
            self.get_audio_feat_path()
        # Initialize acc/gyro feature paths
        if self.args.dataset in ['uci-har', 'extrasensory']:
            self.get_acc_feat_path()
            self.get_gyro_feat_path()
            
    def get_audio_feat_path(self):
        """
        Load audio feature path.
        """
        self.audio_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'audio', 
            self.args.audio_feat, 
            self.args.dataset
        )
        return Path(self.audio_feat_path)
    
    def get_video_feat_path(self):
        """
        Load frame-wise video feature path.
        """
        self.video_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'video', 
            self.args.video_feat, 
            self.args.dataset
        )
        return Path(self.video_feat_path)

    def get_text_feat_path(self):
        """
        Load text feature path.
        """
        self.text_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'text', 
            self.args.text_feat, 
            self.args.dataset
        )
        return Path(self.text_feat_path)
    
    def get_acc_feat_path(self):
        """
        Load acc feature path.
        """
        self.acc_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'acc', 
            self.args.dataset
        )
        return Path(self.acc_feat_path)
    
    def get_gyro_feat_path(self):
        """
        Load gyro feature path.
        """
        self.gyro_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'gyro', 
            self.args.dataset
        )
        return Path(self.gyro_feat_path)

    def get_client_ids(
            self, 
            fold_idx: int=1
        ):
        """
        Load client ids.
        :param fold_idx: fold index
        :return: None
        """
        if self.args.dataset in ["mit51"]:
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.video_feat_path.joinpath(f'alpha{alpha_str}')
        if self.args.dataset in ["uci-har"]:
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.acc_feat_path.joinpath(f'alpha{alpha_str}')
        elif self.args.dataset == "ucf101":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.video_feat_path.joinpath(f'alpha{alpha_str}', f'fold{fold_idx}')
        elif self.args.dataset == "extrasensory":
            data_path = self.acc_feat_path.joinpath(f'fold{fold_idx}')
        elif self.args.dataset == "meld":
            data_path = self.text_feat_path
        self.client_ids = [id.split('.pkl')[0] for id in os.listdir(str(data_path))]
        self.client_ids.sort()
    
    def load_audio_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load audio feature data different applications.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        if self.args.dataset == "ucf101":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.audio_feat_path.joinpath(f'alpha{alpha_str}', f'fold{fold_idx}', f'{client_id}.pkl')
        elif self.args.dataset == "mit51":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.audio_feat_path.joinpath(f'alpha{alpha_str}', f'{client_id}.pkl')
        elif self.args.dataset == "meld":
            data_path = self.audio_feat_path.joinpath(f'{client_id}.pkl')
        
        with open(str(data_path), "rb") as f: 
            data_dict = pickle.load(f)
        return data_dict
    
    def load_video_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load frame-wise video feature data for MMaction applications.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        if self.args.dataset == "ucf101":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.video_feat_path.joinpath(f'alpha{alpha_str}', f'fold{fold_idx}',  f'{client_id}.pkl')
        elif self.args.dataset == "mit51":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.video_feat_path.joinpath(f'alpha{alpha_str}', f'{client_id}.pkl')

        with open(str(data_path), "rb") as f: 
            data_dict = pickle.load(f)
        return data_dict
    
    def load_acc_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load acc feature data for HAR applications.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        if self.args.dataset == "uci-har":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.acc_feat_path.joinpath(f'alpha{alpha_str}', f'{client_id}.pkl')
        elif self.args.dataset == "extrasensory":
            data_path = self.acc_feat_path.joinpath(f'fold{fold_idx}', f'{client_id}.pkl')
        with open(str(data_path), "rb") as f: data_dict = pickle.load(f)
        return data_dict
    
    def load_gyro_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load gyro data for HAR application.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        if self.args.dataset == "uci-har":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.gyro_feat_path.joinpath(f'alpha{alpha_str}', f'{client_id}.pkl')
        elif self.args.dataset == "extrasensory":
            data_path = self.gyro_feat_path.joinpath(f'fold{fold_idx}', f'{client_id}.pkl')
        with open(str(data_path), "rb") as f: data_dict = pickle.load(f)
        return data_dict

    def load_text_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load text feature data.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        if self.args.dataset == "meld":
            data_path = self.text_feat_path.joinpath(f'{client_id}.pkl')
        with open(str(data_path), "rb") as f: 
            data_dict = pickle.load(f)
        return data_dict

    def set_dataloader(
            self, 
            data_a: dict,
            data_b: dict, 
            shuffle: bool=False
        ) -> (DataLoader):
        """
        Set dataloader for training/dev/test.
        :param data_a: modality A data
        :param data_b: modality B data
        :param shuffle: shuffle flag for dataloader, True for training; False for dev and test
        :return: dataloader: torch dataloader
        """
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
        data_ab = MMDatasetGenerator(data_a, data_b, len(data_a))
        if shuffle:
            # we use args input batch size for train, typically set as 16 in FL setup
            dataloader = DataLoader(
                data_ab, 
                batch_size=int(self.args.batch_size), 
                num_workers=0, 
                shuffle=shuffle, 
                collate_fn=collate_mm_fn_padd
            )
        else:
            # we use a larger batch size for validation and testing
            dataloader = DataLoader(
                data_ab, 
                batch_size=64, 
                num_workers=0, 
                shuffle=shuffle, 
                collate_fn=collate_mm_fn_padd
            )
        return dataloader
    
    def load_sim_dict(self, fold_idx: int=1):
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
        elif self.args.dataset in ["mit51", "meld", "uci-har"]:
            data_path = Path(self.args.data_dir).joinpath('simulation_feature',
                                                          self.args.dataset,
                                                          f'{self.setting_str}.pkl')

        with open(str(data_path), "rb") as f: 
            self.sim_data = pickle.load(f)
    
    def get_simulation_setting(self, alpha=None):
        """
        Load get simulation setting string.
        :param alpha: alpha in manual split
        :return: None
        """
        self.setting_str = ''
        # 1. missing modality
        if self.args.missing_modality == True:
            self.setting_str += 'mm'+str(self.args.missing_modailty_rate).replace('.', '')
        # 2. label nosiy
        if self.args.label_nosiy == True:
            if len(self.setting_str) != 0: self.setting_str += '_'
            self.setting_str += 'ln'+str(self.args.label_nosiy_level).replace('.', '')
        # 3. missing labels
        if self.args.missing_label == True:
            if len(self.setting_str) != 0: self.setting_str += '_'
            self.setting_str += 'ml'+str(self.args.missing_label_rate).replace('.', '')
        # 4. alpha for manual split
        if len(self.setting_str) != 0:
            if alpha is not None:
                alpha_str = str(self.args.alpha).replace('.', '')
                self.setting_str += f'_alpha{alpha_str}'