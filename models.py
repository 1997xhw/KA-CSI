import torch
import torch.nn as nn
import torch.nn.functional as F
from mcat import MCAT  # 请确保你已经将 mcat.py 中内容改为 PyTorch 实现
from position_encoding import GRE,Gaussian_Position
from transformer_encoder import Transformer  # 改名为 Transformer，原为 Transfomer（有拼写错误）

class TwoStreamModel(nn.Module):
    def __init__(self, hlayers, vlayers, hheads, vheads, K, sample, num_class, maxlen, embed_dim):
        super(TwoStreamModel, self).__init__()
        self.sample = sample
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.transformer = MCAT(embed_dim, hlayers, hheads, maxlen//2)
        self.pos_encoding = GRE(embed_dim, maxlen//2, K)
        self.pos_encoding_v = GRE(self.maxlen, 30, K)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.use_vertical = vlayers > 0

        self.kernel_num = 128
        self.kernel_num_v = 16
        self.filter_sizes = [20, 40]
        self.filter_sizes_v = [2, 4]

        if self.use_vertical:
            self.v_transformer = Transformer(self.maxlen, vlayers, vheads)
            output_dim = self.kernel_num * len(self.filter_sizes) + self.kernel_num_v * len(self.filter_sizes_v)
            self.dense = torch.nn.Linear(
                self.kernel_num * len(self.filter_sizes) + self.kernel_num_v * len(self.filter_sizes_v), num_class)
        else:
            self.v_transformer = None
            output_dim = self.kernel_num * len(self.filter_sizes)
            self.dense = torch.nn.Linear(90, num_class)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(output_dim, num_class)
        self.feature_proj = nn.Linear(output_dim, 256)  # 将 [B, 288] 投影为 [B, 256]

        self.encoders = []
        self.encoders_v = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = f"encoder_{i}"
            self.__setattr__(enc_attr_name,
                             torch.nn.Conv1d(in_channels=embed_dim,
                                             out_channels=self.kernel_num,
                                             kernel_size=filter_size).to('cuda'))
            self.encoders.append(self.__getattr__(enc_attr_name))

        for i, filter_size in enumerate(self.filter_sizes_v):
            enc_attr_name_v = f"encoder_v_{i}"
            self.__setattr__(enc_attr_name_v,
                             torch.nn.Conv1d(in_channels=self.maxlen,
                                             out_channels=self.kernel_num_v,
                                             kernel_size=filter_size).to('cuda'))
            self.encoders_v.append(self.__getattr__(enc_attr_name_v))

    def _aggregate(self, o, v=None):
        enc_outs = []
        enc_outs_v = []

        for encoder in self.encoders:
            f_map = encoder(o.transpose(-1, -2))
            enc_ = F.relu(f_map)
            k_h = enc_.size()[-1]
            enc_ = F.max_pool1d(enc_, kernel_size=k_h)
            enc_ = enc_.squeeze(dim=-1)
            enc_outs.append(enc_)

        encoding = self.dropout(torch.cat(enc_outs, 1))
        q_re = F.relu(encoding)

        if self.v_transformer is not None:
            for encoder in self.encoders_v:
                f_map = encoder(v.transpose(-1, -2))
                enc_ = F.relu(f_map)
                k_h = enc_.size()[-1]
                enc_ = F.max_pool1d(enc_, kernel_size=k_h)
                enc_ = enc_.squeeze(dim=-1)
                enc_outs_v.append(enc_)

            encoding_v = self.dropout(torch.cat(enc_outs_v, 1))
            v_re = F.relu(encoding_v)
            q_re = torch.cat((q_re, v_re), dim=1)

        return q_re

    def forward(self, data):
        d1 = data.size(0)  #B
        d3 = data.size(2)  #D

        x = data.unsqueeze(-2)
        x = data.view(d1, -1, self.sample, d3)
        x = torch.sum(x, dim=-2).squeeze(-2)
        x = torch.div(x, self.sample)

        x = self.pos_encoding(x)
        x = self.transformer(x)

        if self.use_vertical:
            y = data.view(-1, self.maxlen, self.embed_dim//30, 30)
            y = torch.sum(y, dim=-2).squeeze(-2)
            y = y.transpose(-1, -2)
            y = self.v_transformer(y)
            re = self._aggregate(x, y)
            predict = self.softmax(self.dense(re))
        else:
            re = self._aggregate(x)
            predict = self.softmax(self.dense2(re))

        # return self.feature_proj(re)
        return predict

