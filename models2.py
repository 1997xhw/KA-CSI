import torch
import torch.nn as nn
import torch.nn.functional as F
from mcat import MCAT  # 请确保你已经将 mcat.py 中内容改为 PyTorch 实现
from position_encoding import GRE,Gaussian_Position
from transformer_encoder import Transformer  # 改名为 Transformer，原为 Transfomer（有拼写错误）

class TwoStreamModel(nn.Module):
    def __init__(self, hlayers, vlayers, hheads, vheads, K, sample, num_class, maxlen):
        super(TwoStreamModel, self).__init__()
        self.sample = sample
        self.maxlen = maxlen

        self.transformer = MCAT(270, hlayers, hheads, 500)
        self.pos_encoding = Gaussian_Position(270, 500, K)
        self.pos_encoding_v = Gaussian_Position(2000, 30, K)

        self.use_vertical = vlayers > 0

        self.kernel_num = 128
        self.kernel_num_v = 16
        self.filter_sizes = [20, 40]
        self.filter_sizes_v = [2, 4]

        if self.use_vertical:
            self.v_transformer = Transformer(2000, vlayers, vheads)
            output_dim = self.kernel_num * len(self.filter_sizes) + self.kernel_num_v * len(self.filter_sizes_v)
        else:
            self.v_transformer = None
            output_dim = self.kernel_num * len(self.filter_sizes)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(output_dim, num_class)

        self.encoders = nn.ModuleList([
            nn.Conv1d(in_channels=270, out_channels=self.kernel_num, kernel_size=fs, padding=fs // 2)
            for fs in self.filter_sizes
        ])
        self.encoders_v = nn.ModuleList([
            nn.Conv1d(in_channels=30, out_channels=self.kernel_num_v, kernel_size=fs, padding=fs // 2)
            for fs in self.filter_sizes_v
        ])

    def _aggregate(self, o, v=None):
        o = o.permute(0, 2, 1)  # [B, D, T]
        enc_outs = []
        for conv in self.encoders:
            x = self.relu(conv(o))
            x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
            enc_outs.append(x)
        q_repr = torch.cat(enc_outs, dim=1)
        q_repr = self.dropout(q_repr)

        if v is not None and self.v_transformer is not None:
            v = v.permute(0, 2, 1)
            enc_outs_v = []
            for conv in self.encoders_v:
                x = self.relu(conv(v))
                x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
                enc_outs_v.append(x)
            v_repr = torch.cat(enc_outs_v, dim=1)
            v_repr = self.dropout(v_repr)
            q_repr = torch.cat([q_repr, v_repr], dim=1)

        return q_repr

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.size()
        x = x.view(B, T // self.sample, self.sample, D).sum(dim=2) / self.sample
        x = self.pos_encoding(x)
        x = self.transformer(x)



        # 垂直流处理（假设 D_hidden=270）:
        # [B, 1000, 270] → Reshape → [B, 1000, 9, 30] → Sum → [B, 1000, 30] → Permute → [B, 30, 1000]
        if self.use_vertical:
            y = x.view(-1, 1000, 9, 30).sum(dim=2)  # reshape + sum over antennas
            y = y.permute(0, 2, 1)  # [B, 30, 1000]
            y = self.v_transformer(y)
            out = self._aggregate(x, y)
        else:
            # 水平流处理:
            # [B, T, D] → (降采样) → [B, T // sample, D] → (MCAT) → [B, T // sample, D_hidden]
            out = self._aggregate(x)

        return self.classifier(out)
