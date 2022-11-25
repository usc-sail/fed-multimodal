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
        d_hid: int=128,         # Hidden Layer size
        n_filters: int=32,      # number of filters
        en_att: bool=False,     # Enable self attention or not
        att_name: str=''        # Attention Name
    ):
        super(MMActionClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        
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
            bidirectional=False
        )

        self.video_rnn = nn.GRU(
            input_size=video_input_dim, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=False
        )
        
        # Attention modules
        if self.att_name == "multihead":
            self.audio_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid, 
                num_heads=4, 
                dropout=self.dropout_p
            )
            
            self.video_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid, 
                num_heads=4, 
                dropout=self.dropout_p
            )
        elif self.att_name == "additive":
            self.audio_att = AdditiveAttention(
                d_hid=d_hid, 
                d_att=128
            )
            self.video_att = AdditiveAttention(
                d_hid=d_hid, 
                d_att=128
            )
        elif self.att_name == "base":
            self.audio_att = BaseSelfAttention(
                d_hid=d_hid
            )
            self.video_att = BaseSelfAttention(
                d_hid=d_hid
            )
        elif self.att_name == "hirarchical":
            self.att = HirarchicalAttention(
                d_hid=d_hid
            )
        
        # Projection head
        self.audio_proj = nn.Linear(d_hid, 64)
        self.video_proj = nn.Linear(d_hid, 64)

        # Projection head
        self.init_weight()

        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
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

    def forward(self, x_audio, x_video, len_a, len_v):
        # 1. Conv forward
        x_audio = self.audio_conv(x_audio)
        # 2. Rnn forward
        x_audio, _ = self.audio_rnn(x_audio) # [BxT_axD]
        x_video, _ = self.video_rnn(x_video) # [BxT_vxD]
        # 3. Attention
        if self.en_att:
            if self.att_name == 'multihead':
                x_audio, _ = self.audio_att(x_audio, x_audio, x_audio)
                x_video, _ = self.video_att(x_video, x_video, x_video)
                # 4. Average pooling
                x_audio = torch.mean(x_audio, axis=1)
                x_video = torch.mean(x_video, axis=1)
            elif self.att_name == 'additive':
                # get attention output
                x_audio = self.audio_att(x_audio, x_audio, x_audio, len_a)
                x_video = self.video_att(x_video, x_video, x_video, len_v)
            elif self.att_name == 'base':
                # get attention output
                x_audio = self.audio_att(x_audio)
                x_video = self.video_att(x_video, len_v)
        else:
            # 4. Average pooling
            x_audio = torch.mean(x_audio, axis=1)
            x_video = torch.mean(x_video, axis=1)

        # 5. Projection
        x_audio = self.audio_proj(x_audio)
        x_video = self.video_proj(x_video)
           
        # 6. MM embedding and predict
        x_mm = torch.concat((x_audio, x_video), dim=1)
        preds = self.classifier(x_mm)
        return preds, x_mm


class SERClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int,       # Number of classes 
        audio_input_dim: int,   # Audio data input dim
        text_input_dim: int,    # Text data input dim
        d_hid: int=64,          # Hidden Layer size
        n_filters: int=32,      # number of filters
        en_att: bool=False,     # Enable self attention or not
        att_name: str=''        # Attention Name
    ):
        super(SERClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        
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
            bidirectional=False
        )

        self.text_rnn = nn.GRU(
            input_size=text_input_dim, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=False
        )

        # Self attention module
        if self.att_name == "multihead":
            self.audio_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid, 
                num_heads=4, 
                dropout=self.dropout_p
            )
            self.text_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid, 
                num_heads=4, 
                dropout=self.dropout_p
            )
        elif self.att_name == "base":
            self.audio_att = BaseSelfAttention(
                d_hid=d_hid
            )
            self.text_att = BaseSelfAttention(
                d_hid=d_hid
            )
        
        # Projection head
        self.audio_proj = nn.Linear(d_hid, 64)
        self.text_proj = nn.Linear(d_hid, 64)
        self.init_weight()

        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
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

    def forward(self, x_audio, x_text, l_a, l_b):
        # 1. Conv forward
        x_audio = self.audio_conv(x_audio)
        # 2. Rnn forward
        x_audio, _ = self.audio_rnn(x_audio)
        x_text, _ = self.text_rnn(x_text)
        # 3. Attention
        if self.en_att:
            if self.att_name == 'multihead':
                x_audio, _ = self.audio_att(x_audio, x_audio, x_audio)
                x_text, _ = self.text_att(x_text, x_text, x_text)
                # 4. Average pooling
                x_audio = torch.mean(x_audio, axis=1)
                x_text = torch.mean(x_text, axis=1)
            elif self.att_name == 'base':
                # get attention output
                x_audio = self.audio_att(x_audio)
                x_text = self.text_att(x_text, l_b)
        else:
            # 4. Average pooling
            x_audio = torch.mean(x_audio, axis=1)
            x_text = torch.mean(x_text, axis=1)
        # 5. Projection
        x_audio = self.audio_proj(x_audio)
        x_text = self.text_proj(x_text)
        # 6. MM embedding and predict
        x_mm = torch.concat((x_audio, x_text), dim=1)
        preds = self.classifier(x_mm)
        return preds, x_mm


class HARClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int,       # Number of classes 
        acc_input_dim: int,     # Acc data input dim
        gyro_input_dim: int,    # Gyro data input dim
        d_hid: int=128,         # Hidden Layer size
        n_filters: int=32,      # number of filters
        en_att: bool=False,     # Enable self attention or not
        att_name: str=''        # Attention Name
    ):
        super(HARClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        
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
            bidirectional=False
        )

        self.gyro_rnn = nn.GRU(
            input_size=n_filters*4, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=False
        )

        # Self attention module
        if self.att_name == "multihead":
            self.acc_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid, 
                num_heads=4, 
                dropout=self.dropout_p
            )
            
            self.gyro_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid, 
                num_heads=4, 
                dropout=self.dropout_p
            )
        elif self.att_name == "base":
            self.acc_att = BaseSelfAttention(d_hid=d_hid)
            self.gyro_att = BaseSelfAttention(d_hid=d_hid)
        self.init_weight()

        # Projection head
        self.acc_proj = nn.Linear(d_hid, 64)
        self.gyro_proj = nn.Linear(d_hid, 64)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
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

    def forward(self, x_acc, x_gyro, l_a, l_b):
        # 1. Conv forward
        x_acc = self.acc_conv(x_acc)
        x_gyro = self.gyro_conv(x_gyro)
        # 2. Rnn forward
        x_acc, _ = self.acc_rnn(x_acc)
        x_gyro, _ = self.gyro_rnn(x_gyro)

        # 3. Attention
        if self.en_att:
            if self.att_name == 'multihead':
                x_acc, _ = self.acc_att(x_acc, x_acc, x_acc)
                x_gyro, _ = self.gyro_att(x_gyro, x_gyro, x_gyro)
                # 4. Average pooling
                x_acc = torch.mean(x_acc, axis=1)
                x_gyro = torch.mean(x_gyro, axis=1)
            elif self.att_name == 'base':
                # get attention output
                x_acc = self.acc_att(x_acc)
                x_gyro = self.gyro_att(x_gyro)
        else:
            # 4. Average pooling
            x_acc = torch.mean(x_acc, axis=1)
            x_gyro = torch.mean(x_gyro, axis=1)

        # 5. Projection
        x_acc = self.acc_proj(x_acc)
        x_gyro = self.gyro_proj(x_gyro)
        # 6. MM embedding and predict
        x_mm = torch.concat((x_acc, x_gyro), dim=1)
        preds = self.classifier(x_mm)
        return preds, x_mm


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
        return preds, x_mm


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
    
    
def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
    

class AdditiveAttention(nn.Module):
    def __init__(
        self, 
        d_hid:  int=64, 
        d_att:  int=256
    ):
        super().__init__()

        self.query_proj = nn.Linear(d_hid, d_att, bias=False)
        self.key_proj = nn.Linear(d_hid, d_att, bias=False)
        self.bias = nn.Parameter(torch.rand(d_att).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(d_att, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self, 
        query: Tensor,
        key: Tensor, 
        value: Tensor,
        valid_lens: Tensor
    ):
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        # attn = F.softmax(score, dim=-1)
        attn = masked_softmax(scores, valid_lens)
        attn = self.dropout(attn)
        output = torch.bmm(attn.unsqueeze(1), value)
        return output
    

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            pdb.set_trace()
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class HirarchicalAttention(nn.Module):
    '''
    ref: Hierarchical Attention Networks
    '''

    def __init__(self, d_hid: int):
        super(HirarchicalAttention, self).__init__()
        self.w_linear = nn.Linear(d_hid, d_hid)
        self.u_w = nn.Linear(d_hid, 1, bias=False)

    def forward(self, input: torch.Tensor):
        u_it = torch.tanh(self.w_linear(input))
        a_it = torch.softmax(self.u_w(u_it), dim=1)
        s_i = input * a_it
        return s_i


class HirarchicalAttention(nn.Module):
    '''
    ref: Hierarchical Attention Networks
    '''

    def __init__(self, d_hid: int):
        super(HirarchicalAttention, self).__init__()
        self.w_linear = nn.Linear(d_hid, d_hid)
        self.u_w = nn.Linear(d_hid, 1, bias=False)

    def forward(self, input: torch.Tensor):
        u_it = torch.tanh(self.w_linear(input))
        a_it = torch.softmax(self.u_w(u_it), dim=1)
        s_i = input * a_it
        return s_i
    

class BaseSelfAttention(nn.Module):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8421023
    def __init__(
        self, 
        d_hid:  int=64
    ):
        super().__init__()
        self.att_fc1 = nn.Linear(d_hid, 1)
        self.att_pool = nn.Tanh()
        self.att_fc2 = nn.Linear(1, 1)

    def forward(
        self,
        x: Tensor,
        val_l=None
    ):
        att = self.att_pool(self.att_fc1(x))
        att = self.att_fc2(att).squeeze(-1)
        if val_l is not None:
            for idx in range(len(val_l)):
                att[idx, val_l[idx]:] = -1e6
        att = torch.softmax(att, dim=1)
        x = (att.unsqueeze(2) * x).sum(axis=1)
        return x