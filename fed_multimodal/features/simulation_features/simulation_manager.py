import json
import glob
import torch
import pickle
import random
import pdb, os
import numpy as np
import os.path as osp

from PIL import Image
from tqdm import tqdm
from pathlib import Path


class SimulationManager():
    def __init__(self, args: dict):
        self.args = args
        
    def fetch_partition(
        self, 
        fold_idx=1, 
        alpha=0.5,
        ext='json'
    ):
        # reading partition
        if self.args.dataset == "ucf101":
            alpha_str = str(alpha).replace('.', '')
            partition_path = Path(self.args.output_dir).joinpath(
                "partition", 
                self.args.dataset, 
                f'fold{fold_idx}', 
                f'partition_alpha{alpha_str}.{ext}'
            )
        elif self.args.dataset in ["mit10", "mit51", "uci-har", "crisis-mmd", "hateful_memes"]:
            alpha_str = str(alpha).replace('.', '')
            partition_path = Path(self.args.output_dir).joinpath(
                "partition", 
                self.args.dataset, 
                f'partition_alpha{alpha_str}.{ext}'
            )
        elif self.args.dataset in ["meld", "ptb-xl"]:
            partition_path = Path(self.args.output_dir).joinpath(
                "partition", 
                self.args.dataset, 
                f'partition.{ext}'
            )
        elif self.args.dataset in ["crema_d", "ku-har"]:
            partition_path = Path(self.args.output_dir).joinpath(
                "partition", 
                self.args.dataset,
                f'fold{fold_idx}',
                f'partition.{ext}'
            )
        else:
            raise ValueError(
                f'Dataset not found {self.args.dataset}'
            )

        if ext == "pkl":
            with open(str(partition_path), "rb") as f: 
                partition_dict = pickle.load(f)
        else:
            with open(str(partition_path), "r") as f: 
                partition_dict = json.load(f)

        return partition_dict
    
    def simulate_missing_modality(self, seed):
        np.random.seed(seed)
        return np.random.binomial(
            size=1, 
            n=1, 
            p=self.args.missing_modailty_rate
        )[0]
    
    def simulate_missing_label(self, seed, size):
        np.random.seed(seed)
        return np.random.binomial(
            size=size, 
            n=1, 
            p=self.args.missing_label_rate
        )
    
    def label_noise_matrix(
        self, 
        seed, 
        class_num=51
    ):
        # create matrix for each user
        np.random.seed(seed)
        noisy_level = self.args.label_nosiy_level
        sparse_level = 0.4
        prob_matrix = [1-noisy_level] * class_num * class_num
        sparse_elements = np.random.choice(
            class_num*class_num, 
            round(class_num*(class_num-1)*sparse_level)
        )
        for idx in range(len(sparse_elements)):
            while sparse_elements[idx]%(class_num+1) == 0:
                sparse_elements[idx] = np.random.choice(class_num*class_num, 1)
            prob_matrix[sparse_elements[idx]] = 0

        available_spots = np.argwhere(np.array(prob_matrix) == 1 - noisy_level)
        for idx in range(class_num):
            available_spots = np.delete(available_spots, np.argwhere(available_spots == idx*(class_num+1)))

        for idx in range(class_num):
            row = prob_matrix[idx*4:(idx*4)+4]
            if len(np.where(np.array(row) == 1 - noisy_level)[0]) == 2:
                unsafe_points = np.where(np.array(row) == 1 - noisy_level)[0]
                unsafe_points = np.delete(unsafe_points, np.where(np.array(unsafe_points) == idx*(class_num+1))[0])
                available_spots = np.delete(available_spots, np.argwhere(available_spots == unsafe_points[0]))
            if np.sum(row) == 1 - noisy_level:
                zero_spots = np.where(np.array(row) == 0)[0]
                prob_matrix[zero_spots[0] + idx * 4], prob_matrix[available_spots[0]] = prob_matrix[available_spots[0]], prob_matrix[zero_spots[0] + idx * 4]
                available_spots = np.delete(available_spots, 0) 

        prob_matrix = np.reshape(prob_matrix, (class_num, class_num))
        
        for idx in range(len(prob_matrix)):
            zeros = np.count_nonzero(prob_matrix[idx]==0)
            if class_num-zeros-1 == 0:
                prob_element = 0
            else:
                prob_element = (noisy_level) / (class_num-zeros-1)
            prob_matrix[idx] = np.where(prob_matrix[idx] == 1-noisy_level, prob_element, prob_matrix[idx])
            prob_matrix[idx][idx] = 1-noisy_level
            
        return prob_matrix
    
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
        
    def simulation(
        self, 
        data_dict: dict, 
        seed: int, 
        class_num: int=51
    ) -> (dict):
        # 1. simulate modality missing
        if self.args.missing_modality == True:
            modality_a_missing = int(self.simulate_missing_modality(seed=seed))
            modality_b_missing = int(self.simulate_missing_modality(seed=seed*2))
        else:
            modality_a_missing = 0
            modality_b_missing = 0
            
        # 2. generate label noise matrix for later
        if self.args.label_nosiy == True:
            self.prob_matrix = self.label_noise_matrix(
                seed=seed, 
                class_num=class_num
            )
        
        # 3. missing label simulation
        if self.args.missing_label == True:
            missing_label_array = self.simulate_missing_label(
                seed=seed, 
                size=len(data_dict)
            )
        
        # 2. simulate others
        for idx in range(len(data_dict)):
            # 2.1 simulate label noise
            if self.args.label_nosiy == True:
                orginal_label = data_dict[idx][2]
                np.random.seed(seed)
                if self.args.dataset == 'hateful_memes':
                    change_status = np.random.binomial(size=1, n=1, p=self.args.label_nosiy_level)[0]
                    new_label = orginal_label
                    if change_status == 1:
                        new_label = 1 - orginal_label
                else:
                    new_label = np.random.choice(
                        class_num, 
                        p=self.prob_matrix[orginal_label]
                    )
            else:
                new_label = data_dict[idx][2]
            
            # 2.2 simulate missing label
            if self.args.missing_label == True:
                missing_label = int(missing_label_array[idx])
            else:
                missing_label = 0
            
            # 2.3 simulation feature vector
            # [missing_modalityA, missing_modalityB, new_label, missing_label]
            data_dict[idx].append([modality_a_missing, modality_b_missing, new_label, missing_label])
            
        return data_dict