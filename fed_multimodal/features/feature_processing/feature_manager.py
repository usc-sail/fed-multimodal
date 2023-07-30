# Author: Tiantian Feng
# USC SAIL lab, tiantiaf@usc.edu
import glob
import json
import torch
import pickle
import random
import pdb, os
import torchaudio
import numpy as np
import os.path as osp

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from moviepy.editor import *
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from transformers import AlbertTokenizer, AlbertModel
from transformers import MobileBertTokenizer, MobileBertModel


class FeatureManager():
    def __init__(self, args: dict):
        self.args = args
        if 'feature_type' in args: self.initialize_feature_module()
        
    def initialize_feature_module(self):
        # device = gpu or cpu
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available(): print("GPU available, use GPU")
        
        # load models
        if self.args.feature_type == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=True)
            self.model.classifier = self.model.classifier[:-1]
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # image transform
            self.img_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif self.args.feature_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            # Load pre-trained model (weights)
            self.model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        
            # Put the model in "evaluation" mode, meaning feed-forward operation.
            self.model = self.model.to(self.device)
            self.model.eval()
        elif self.args.feature_type == "mobilebert":
            self.tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
            self.model = MobileBertModel.from_pretrained("google/mobilebert-uncased")
            # Put the model in "evaluation" mode, meaning feed-forward operation.
            self.model = self.model.to(self.device)
            self.model.eval()
            
    def extract_frame_features(
        self, 
        video_id: str, 
        label_str: str,
        max_len: int=-1,
        split=None
    ) -> (np.array):
        """
        Extract the framewise feature from video streams
        :param video_id: video id
        :param label_str: label string
        :param max_len: max len of the features
        :return: return features
        """
        if split is None:
            video_path = Path(self.args.raw_data_dir).joinpath(
                self.args.dataset,
                'rawframes', 
                label_str, 
                video_id
            )
        else: 
            video_path = Path(self.args.raw_data_dir).joinpath(
                self.args.dataset,
                'rawframes', 
                split, 
                label_str, 
                video_id
            )
        rawframes = os.listdir(video_path)
        rawframes.sort()
        if self.args.dataset == "ucf101":
            # downsample to 1 sec per frame
            rawframes = rawframes[::25]
        elif self.args.dataset == "ucf51":
            # downsample to every 10 frames
            rawframes = rawframes[::10]
        
        input_data_list = list()
        for rawframe in rawframes:
            rawframe_path = Path.joinpath(video_path.joinpath(rawframe))
            input_image = Image.open(rawframe_path)
            input_tensor = self.img_transform(input_image)
            input_data_list.append(input_tensor.detach().cpu().numpy())
        
        with torch.no_grad():
            input_data = torch.Tensor(np.array(input_data_list)).to(self.device)
            if len(input_data) == 0: return None
            features = self.model(input_data).detach().cpu().numpy()
        if max_len != -1: features = features[:max_len]
        return features
    
    def extract_img_features(
        self, 
        img_path: str
    ) -> (np.array):
        """
        Extract the framewise feature from image
        :param img_path: image path
        :return: return features
        """
        input_image = Image.open(img_path).convert('RGB')
        input_tensor = self.img_transform(input_image)
        
        with torch.no_grad():
            input_data = input_tensor.to(self.device).unsqueeze(dim=0)
            features = self.model(input_data).detach().cpu().numpy()
        return features
    
    def extract_frame_features_ser(
        self, 
        video_path: str,
        max_len: int=10
    ) -> (np.array):
        """
        Extract the framewise feature from video streams
        :param video_path: video paths
        :return: return features
        """
        try:
            clip = VideoFileClip(str(video_path))
        except:
            raise FileNotFoundError
        
        # read fps and total frames
        fps = int(clip.fps)
        nframes = clip.reader.nframes
        # pdb.set_trace()
        
        # read fps and total frames read frames data
        input_data_list = list()
        for frame in clip.iter_frames():
            input_tensor = self.img_transform(Image.fromarray(frame))
            input_data_list.append(input_tensor.detach().cpu().numpy())
        # just use one frame per second
        input_data_list = input_data_list[:-2]
        input_data_list = input_data_list[::fps]
        
        # extract pretrained feature
        with torch.no_grad():
            input_data = torch.Tensor(np.array(input_data_list)).to(self.device)
            if len(input_data) == 0: return None
            features = self.model(input_data).detach().cpu().numpy()
        if max_len != -1: features = features[:max_len]
        return features
    
    def extract_mfcc_features(
        self, 
        audio_path: str, 
        label_str: str='',
        frame_length: int=40,
        frame_shift:  int=20,
        max_len: int=-1,
        en_znorm: bool=True
    ) -> (np.array):
        """Extract the mfcc feature from audio streams."""
        audio, sr = torchaudio.load(str(audio_path))
        if audio.shape[0] != 1:
            audio = torch.mean(audio, dim=0).unsqueeze(0)
        if sr != 16000:
            transform_model = torchaudio.transforms.Resample(sr, 16000)
            audio = transform_model(audio)
        features = torchaudio.compliance.kaldi.fbank(
                    waveform=torch.Tensor(torch.Tensor(audio)),
                    frame_length=frame_length, 
                    frame_shift=frame_shift,
                    num_mel_bins=80,
                    window_type="hamming")
        features = features.detach().cpu().numpy()
        if en_znorm:
            features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-5)
        if max_len != -1: features = features[:max_len]
        return features
    
    def fetch_partition(
        self, 
        fold_idx: int=1, 
        alpha: float=0.5,
        file_ext: str="json"
    ) -> (dict):
        """
        Read partition file
        :param fold_idx: fold index
        :param alpha: manual split non-iidness, the lower, the more skewed distribution is
        :return: return partition_dict
        """
        # reading partition
        alpha_str = str(alpha).replace('.', '')
        if self.args.dataset == "ucf101":
            partition_path = Path(self.args.output_dir).joinpath(
                "partition", 
                self.args.dataset, 
                f'fold{fold_idx}', 
                f'partition_alpha{alpha_str}.{file_ext}'
            )
        elif self.args.dataset in ["extrasensory", "extrasensory_watch", "crema_d", "ku-har"]:
            partition_path = Path(self.args.output_dir).joinpath(
                "partition", 
                self.args.dataset, 
                f'fold{fold_idx}', 
                f'partition.{file_ext}'
            )
        elif self.args.dataset in ["mit10", "mit51", "uci-har", "hateful_memes", "crisis-mmd"]:
            partition_path = Path(self.args.output_dir).joinpath(
                "partition", 
                self.args.dataset, 
                f'partition_alpha{alpha_str}.{file_ext}'
            )
        elif self.args.dataset in ["meld", "ptb-xl"]:
            partition_path = Path(self.args.output_dir).joinpath(
                "partition", 
                self.args.dataset, 
                f'partition.{file_ext}'
            )
        # raise error if file not exists
        if Path.exists(partition_path) == False: 
            raise FileNotFoundError(
                'No partition file exists at the location specified'
            )
        # read file based on pkl, json
        if file_ext == "pkl":
            with open(str(partition_path), "rb") as f:  
                partition_dict = pickle.load(f)
        else:
            with open(str(partition_path), "r") as f:  
                partition_dict = json.load(f)
        return partition_dict

    def extract_text_feature(
        self, 
        input_str: str
    ) -> (np.array):
        """
        Extract features
        :param input_str: input string
        :return: return embeddings
        """
        with torch.no_grad():
            inputs = self.tokenizer(input_str, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state.detach().cpu().numpy()[0]
        return features