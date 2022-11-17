#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tiantian
"""
import pdb
import torch
import numpy as np
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

# typing import
from typing import Dict, Iterable, Optional


class MMActionClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int,       # Number of classes 
        audio_input_dim: int,   # Audio feature input dim
        video_input_dim: int,   # Frame-wise video feature input dim
        d_hid: int=64,          # Hidden Layer size
        n_filters: int=32,      # number of filters
        en_att: bool=False      # Enable self attention or not
    ):
        super(MMActionClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        
        # Conv Encoder module
        self.audio_conv = Conv1dEncoder(
            input_dim=audio_input_dim, 
            n_filters=n_filters, 
            dropout=self.dropout_p, 
        )
        
        # RNN module
        self.audio_rnn = nn.GRU(
            input_size=n_filters*4, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=True
        )

        self.video_rnn = nn.GRU(
            input_size=video_input_dim, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=True
        )

        # Self attention module
        self.audio_att = SelfAttention(d_hid=d_hid, d_att=256, n_head=4)
        self.video_att = SelfAttention(d_hid=d_hid, d_att=256, n_head=4)
        
        # Projection head
        self.audio_proj = nn.Linear(d_hid*2, 128)
        self.video_proj = nn.Linear(d_hid*2, 128)
        self.init_weight()

        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(128, num_classes)
        )

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x_audio, x_video):
        # 1. Conv forward
        x_audio = self.audio_conv(x_audio)
        # 2. Rnn forward
        x_audio, _ = self.audio_rnn(x_audio)
        x_video, _ = self.video_rnn(x_video)
        # 3. Attention
        if self.en_att:
            x_audio = self.audio_att(x_audio)
            x_video = self.video_att(x_video)
        # 4. Average pooling
        x_audio = torch.mean(x_audio, axis=1)
        x_video = torch.mean(x_video, axis=1)
        # 5. Projection
        x_audio = self.audio_proj(x_audio)
        x_video = self.video_proj(x_video)
        # 6. MM embedding and predict
        x_mm = torch.concat((x_audio, x_video), dim=1)
        preds = self.classifier(x_mm)
        return preds


class SERClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int,       # Number of classes 
        audio_input_dim: int,   # Audio data input dim
        text_input_dim: int,    # Text data input dim
        d_hid: int=64,          # Hidden Layer size
        n_filters: int=32,      # number of filters
        en_att: bool=False      # Enable self attention or not
    ):
        super(SERClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        
        # Conv Encoder module
        self.audio_conv = Conv1dEncoder(
            input_dim=audio_input_dim, 
            n_filters=n_filters, 
            dropout=self.dropout_p, 
        )
        
        # RNN module
        self.audio_rnn = nn.GRU(
            input_size=n_filters*4, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=True
        )

        self.text_rnn = nn.GRU(
            input_size=text_input_dim, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=True
        )
        # Self attention module
        self.audio_att = SelfAttention(d_hid=d_hid, d_att=256, n_head=4)
        self.text_att = SelfAttention(d_hid=d_hid, d_att=256, n_head=4)
        
        # Projection head
        self.audio_proj = nn.Linear(d_hid*2, 128)
        self.text_proj = nn.Linear(d_hid*2, 128)
        self.init_weight()

        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x_audio, x_text):
        # 1. Conv forward
        x_audio = self.audio_conv(x_audio)
        # 2. Rnn forward
        x_audio, _ = self.audio_rnn(x_audio)
        x_text, _ = self.text_rnn(x_text)
        # 3. Attention
        if self.en_att:
            x_audio = self.audio_att(x_audio)
            x_text = self.text_att(x_text)
        # 4. Average pooling
        x_audio = torch.mean(x_audio, axis=1)
        x_text = torch.mean(x_text, axis=1)
        # 5. Projection
        x_audio = self.audio_proj(x_audio)
        x_text = self.text_proj(x_text)
        # 6. MM embedding and predict
        x_mm = torch.concat((x_audio, x_text), dim=1)
        preds = self.classifier(x_mm)
        return preds


class HARClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int,       # Number of classes 
        acc_input_dim: int,     # Acc data input dim
        gyro_input_dim: int,    # Gyro data input dim
        d_hid: int=64,          # Hidden Layer size
        n_filters: int=32,      # number of filters
        en_att: bool=False      # Enable self attention or not
    ):
        super(HARClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        
        # Conv Encoder module
        self.acc_conv = Conv1dEncoder(
            input_dim=acc_input_dim, 
            n_filters=n_filters, 
            dropout=self.dropout_p, 
        )
        
        self.gyro_conv = Conv1dEncoder(
            input_dim=acc_input_dim, 
            n_filters=n_filters, 
            dropout=self.dropout_p, 
        )
        
        # RNN module
        self.acc_rnn = nn.GRU(
            input_size=n_filters*4, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=True
        )

        self.gyro_rnn = nn.GRU(
            input_size=n_filters*4, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=True
        )

        # Self attention module
        self.acc_att = SelfAttention(d_hid=d_hid, d_att=256, n_head=4)
        self.gyro_att = SelfAttention(d_hid=d_hid, d_att=256, n_head=4)
        self.init_weight()

        # Projection head
        self.acc_proj = nn.Linear(d_hid*2, 64)
        self.gyro_proj = nn.Linear(d_hid*2, 64)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x_acc, x_gyro):
        # 1. Conv forward
        x_acc = self.acc_conv(x_acc)
        x_gyro = self.gyro_conv(x_gyro)
        # 2. Rnn forward
        x_acc, _ = self.acc_rnn(x_acc)
        x_gyro, _ = self.gyro_rnn(x_gyro)
        # 3. Attention
        if self.en_att:
            x_acc = self.acc_att(x_acc)
            x_gyro = self.gyro_att(x_gyro)
        # 4. Average pooling
        x_acc = torch.mean(x_acc, axis=1)
        x_gyro = torch.mean(x_gyro, axis=1)
        # 5. Projection
        x_acc = self.acc_proj(x_acc)
        x_gyro = self.gyro_proj(x_gyro)
        # 6. MM embedding and predict
        x_mm = torch.concat((x_acc, x_gyro), dim=1)
        preds = self.classifier(x_mm)
        return preds


class ECGClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int,           # Number of classes 
        i_to_avf_input_dim: int,    # 6 lead ecg
        v1_to_v6_input_dim: int,    # v1-v6 ecg
        d_hid: int=64,              # Hidden Layer size
        n_filters: int=32,          # number of filters
        en_att: bool=False          # Enable self attention or not
    ):
        super(ECGClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        
        # Conv Encoder module
        self.i_to_avf_conv = Conv1dEncoder(
            input_dim=i_to_avf_input_dim, 
            n_filters=n_filters, 
            dropout=self.dropout_p, 
        )
        
        self.v1_to_v6_conv = Conv1dEncoder(
            input_dim=v1_to_v6_input_dim, 
            n_filters=n_filters, 
            dropout=self.dropout_p, 
        )
        
        # RNN module
        self.i_to_avf_rnn = nn.GRU(
            input_size=n_filters*4, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=True
        )

        self.v1_to_v6_rnn = nn.GRU(
            input_size=n_filters*4, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=True
        )

        # Self attention module
        self.i_to_avf_att = SelfAttention(d_hid=d_hid, d_att=256, n_head=4)
        self.v1_to_v6_att = SelfAttention(d_hid=d_hid, d_att=256, n_head=4)
        self.init_weight()

        # Projection head
        self.i_to_avf_proj = nn.Linear(d_hid*2, 64)
        self.v1_to_v6_proj = nn.Linear(d_hid*2, 64)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x_i_to_avf, x_v1_to_v6):
        # 1. Conv forward
        x_i_to_avf = self.i_to_avf_conv(x_i_to_avf)
        x_v1_to_v6 = self.v1_to_v6_conv(x_v1_to_v6)
        # 2. Rnn forward
        x_i_to_avf, _ = self.i_to_avf_rnn(x_i_to_avf)
        x_v1_to_v6, _ = self.v1_to_v6_rnn(x_v1_to_v6)
        # 3. Attention
        if self.en_att:
            x_i_to_avf = self.i_to_avf_att(x_i_to_avf)
            x_v1_to_v6 = self.v1_to_v6_att(x_v1_to_v6)
        # 4. Average pooling
        x_i_to_avf = torch.mean(x_i_to_avf, axis=1)
        x_v1_to_v6 = torch.mean(x_v1_to_v6, axis=1)
        # 5. Projection
        x_i_to_avf = self.i_to_avf_proj(x_i_to_avf)
        x_v1_to_v6 = self.v1_to_v6_proj(x_v1_to_v6)
        # 6. MM embedding and predict
        x_mm = torch.concat((x_i_to_avf, x_v1_to_v6), dim=1)
        preds = self.classifier(x_mm)
        return preds


class Conv1dEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int, 
        n_filters: int,
        dropout: float=0.1
    ):
        super().__init__()
        # conv module
        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(n_filters, n_filters*2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(n_filters*2, n_filters*4, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
            self,
            x: Tensor   # shape => [batch_size (B), num_data (T), feature_dim (D)]
        ):
        x = x.float()
        x = x.permute(0, 2, 1)
        # conv1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        # conv2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        # conv3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x

class SelfAttention(nn.Module):
    def __init__(
        self, 
        d_hid:  int=64, 
        d_att:  int=512, 
        n_head: int=8
    ):
        super().__init__()
        self.att_linear1 = nn.Linear(d_hid*2, d_att)
        self.att_pool = nn.Tanh()
        self.att_linear2 = nn.Linear(d_att, n_head)
        
        self.att_mat1 = torch.nn.Parameter(torch.rand(d_att, d_hid*2), requires_grad=True)
        self.att_mat2 = torch.nn.Parameter(torch.rand(n_head, d_att), requires_grad=True)

    def forward(
        self,
        x: Tensor
    ):
        att = self.att_linear1(x)
        att = self.att_pool(att)
        att = self.att_linear2(att)
        att = att.transpose(1, 2)
        att = torch.softmax(att, dim=2)
        x = torch.matmul(att, x)
        return x