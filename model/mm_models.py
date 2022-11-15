#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tiantian
"""
import pdb
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from typing import Dict
from typing import Iterable, Optional
import numpy as np


class MMactionClassifier(nn.Module):
    def __init__(self, num_classes, audio_input_dim, video_input_dim, hidden_size=128):
        super(MMactionClassifier, self).__init__()
        self.dropout_p = 0.25
        self.rnn_dropout = nn.Dropout(self.dropout_p)

        self.audio_rnn = nn.GRU(input_size=128, hidden_size=hidden_size, 
                                num_layers=1, batch_first=True, 
                                dropout=self.dropout_p, bidirectional=True)

        self.video_rnn = nn.GRU(input_size=video_input_dim, hidden_size=hidden_size, 
                                num_layers=1, batch_first=True, 
                                dropout=self.dropout_p, bidirectional=True)

        # conv module
        self.audio_conv = nn.Sequential(
            nn.Conv1d(audio_input_dim, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),
            
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),

            nn.Conv1d(96, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),
        )

        self.init_weight()

        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(hidden_size*2, 128)
        )

        self.video_proj = nn.Sequential(
            nn.Linear(hidden_size*2, 128)
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
        
        # audio
        x_audio = x_audio.float()
        x_audio = x_audio.permute(0, 2, 1)
        x_audio = self.audio_conv(x_audio)
        x_audio = x_audio.permute(0, 2, 1)
        
        x_audio, _ = self.audio_rnn(x_audio)
        x_audio = x_audio[:, 0, :]

        # video
        x_video = x_video.float()
        x_video, _ = self.video_rnn(x_video)
        x_video = x_video[:, 0, :]

        # projection
        x_audio = self.audio_proj(x_audio)
        x_video = self.video_proj(x_video)
        x_mm = torch.concat((x_audio, x_video), dim=1)

        preds = self.classifier(x_mm)
        return preds


class SERClassifier(nn.Module):
    def __init__(self, num_classes, audio_input_dim, text_input_dim, hidden_size=128, att=None):
        super(SERClassifier, self).__init__()
        self.dropout_p = 0.25
        
        self.audio_rnn = nn.GRU(input_size=128, hidden_size=hidden_size, 
                                num_layers=1, batch_first=True, 
                                dropout=self.dropout_p, bidirectional=True)

        self.text_rnn = nn.GRU(input_size=text_input_dim, hidden_size=hidden_size, 
                               num_layers=1, batch_first=True, 
                               dropout=self.dropout_p, bidirectional=True)
        
        # conv module
        self.audio_conv = nn.Sequential(
            nn.Conv1d(audio_input_dim, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),
            
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),

            nn.Conv1d(96, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),
        )

        self.init_weight()

        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(hidden_size*2, 128)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(hidden_size*2, 128)
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
        
        # audio
        x_audio = x_audio.float()
        x_audio = x_audio.permute(0, 2, 1)
        x_audio = self.audio_conv(x_audio)
        x_audio = x_audio.permute(0, 2, 1)
        x_audio, _ = self.audio_rnn(x_audio)
        x_audio = x_audio[:, 0, :]

        # video
        x_text = x_text.float()
        x_text, _ = self.text_rnn(x_text)
        x_text = x_text[:, 0, :]

        # projection
        x_audio = self.audio_proj(x_audio)
        x_text = self.text_proj(x_text)
        x_mm = torch.concat((x_audio, x_text), dim=1)

        preds = self.classifier(x_mm)
        return preds


class HARClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int,       # Number of classes 
        acc_input_dim: int,     # Acc data input dim
        gyro_input_dim: int,    # Acc data input dim
        d_hid: int=64,          # Hidden Layer size
        en_att: bool=False      # Enable self attention or not
    ):
        super(HARClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        
        # Conv Encoder module
        self.acc_conv = Conv1dEncoder(
            input_dim=acc_input_dim, 
            n_filters=32, 
            dropout=self.dropout_p, 
        )
        
        self.gyro_conv = Conv1dEncoder(
            input_dim=acc_input_dim, 
            n_filters=32, 
            dropout=self.dropout_p, 
        )
        
        # RNN module
        self.acc_rnn = nn.GRU(
            input_size=128, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=True
        )

        self.gyro_rnn = nn.GRU(
            input_size=128, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=True
        )

        # Self attention module
        self.acc_att = SelfAttention(d_hid=d_hid, d_att=256, n_head=8)
        self.gyro_att = SelfAttention(d_hid=d_hid, d_att=256, n_head=8)
        self.init_weight()

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        self.acc_proj = nn.Sequential(
            nn.Linear(hidden_size*2, 64)
        )

        self.gyro_proj = nn.Sequential(
            nn.Linear(hidden_size*2, 64)
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
        d_hid: int, 
        d_att: int, 
        n_head: int
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
        
class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype)
        )

class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=4, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        self.dropout = nn.Dropout(p=0.1)
        
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        # pdb.set_trace()
        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv = self.qkv_attention(q, k, v, mask)
        return self.out(wv)

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)
        x = x + self.mlp(self.mlp_ln(x))
        return x
    
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    