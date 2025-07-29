import torch
import torch.nn as nn
import torch.nn.functional as F
from mcat import MCAT  
from position_encoding import GRE, Gaussian_Position, LearnablePositionalEncoding
from transformer_encoder import Transformer  
from kan_grn import KAN_GatesResidualNetwork  # 新增导入



class THATModel(nn.Module):
    def __init__(self, hlayers, vlayers, hheads, vheads, K, sample, num_class, maxlen, embed_dim):
        super(THATModel, self).__init__()
        # 采样数
        self.sample = sample
        # 最大长度
        self.maxlen = maxlen
        # 嵌入维度
        self.embed_dim = embed_dim
        # 水平Transformer（MCAT）
        self.transformer = MCAT(embed_dim, hlayers, hheads, maxlen//2)
        # 水平位置编码
        self.pos_encoding = GRE(embed_dim, maxlen//2, K)
        # 垂直位置编码
        self.pos_encoding_v = GRE(self.maxlen, 30, K)
        # log softmax用于分类
        self.softmax = torch.nn.LogSoftmax(dim=1)
        # 是否使用垂直分支
        self.use_vertical = vlayers > 0

        # 卷积核数量
        self.kernel_num = 128
        self.kernel_num_v = 16
        # 卷积核尺寸
        self.filter_sizes = [20, 40]
        self.filter_sizes_v = [2, 4]

        if self.use_vertical:
            # 垂直Transformer
            self.v_transformer = Transformer(self.maxlen, vlayers, vheads, use_relative=True, max_len=self.maxlen)
            # 输出维度 = 水平卷积输出 + 垂直卷积输出
            output_dim = self.kernel_num * len(self.filter_sizes) + self.kernel_num_v * len(self.filter_sizes_v)
            # 全连接层用于分类
            self.dense = torch.nn.Linear(
                self.kernel_num * len(self.filter_sizes) + self.kernel_num_v * len(self.filter_sizes_v), num_class)
        else:
            self.v_transformer = None
            output_dim = self.kernel_num * len(self.filter_sizes)
            # 这里的90是特征拼接后的维度
            self.dense = torch.nn.Linear(90, num_class)

        # dropout防止过拟合
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        # 分类器
        self.classifier = nn.Linear(output_dim, num_class)
        # 特征投影层，将输出投影到256维
        self.feature_proj = nn.Linear(output_dim, 256)  # 将 [B, 288] 投影为 [B, 256]

        # 卷积编码器列表（水平）
        self.encoders = []
        # 卷积编码器列表（垂直）
        self.encoders_v = []
        # 构建水平卷积编码器
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = f"encoder_{i}"
            self.__setattr__(enc_attr_name,
                             torch.nn.Conv1d(in_channels=embed_dim,
                                             out_channels=self.kernel_num,
                                             kernel_size=filter_size).to('cuda'))
            self.encoders.append(self.__getattr__(enc_attr_name))

        # 构建垂直卷积编码器
        for i, filter_size in enumerate(self.filter_sizes_v):
            enc_attr_name_v = f"encoder_v_{i}"
            self.__setattr__(enc_attr_name_v,
                             torch.nn.Conv1d(in_channels=self.maxlen,
                                             out_channels=self.kernel_num_v,
                                             kernel_size=filter_size).to('cuda'))
            self.encoders_v.append(self.__getattr__(enc_attr_name_v))

    def _aggregate(self, o, v=None):
        """
        特征聚合函数
        o: 水平特征 [B, L, D]
        v: 垂直特征 [B, D, L]（可选）
        """
        enc_outs = []
        enc_outs_v = []

        # 水平卷积特征提取
        for encoder in self.encoders:
            f_map = encoder(o.transpose(-1, -2))  # [B, D, L] -> [B, D, L] 做1D卷积（CNN）
            enc_ = F.relu(f_map)
            k_h = enc_.size()[-1]
            enc_ = F.max_pool1d(enc_, kernel_size=k_h)  # 池化
            enc_ = enc_.squeeze(dim=-1)
            enc_outs.append(enc_)

        # 拼接所有卷积输出
        encoding = self.dropout(torch.cat(enc_outs, 1))
        q_re = F.relu(encoding)

        # 如果有垂直分支，提取垂直特征
        if self.v_transformer is not None and v is not None:
            for encoder in self.encoders_v:
                f_map = encoder(v.transpose(-1, -2))
                enc_ = F.relu(f_map)
                k_h = enc_.size()[-1]
                enc_ = F.max_pool1d(enc_, kernel_size=k_h)
                enc_ = enc_.squeeze(dim=-1)
                enc_outs_v.append(enc_)

            encoding_v = self.dropout(torch.cat(enc_outs_v, 1))
            v_re = F.relu(encoding_v)
            # 水平和垂直特征拼接
            q_re = torch.cat((q_re, v_re), dim=1)

        return q_re

    def forward(self, data):
        """
        前向传播
        data: 输入数据 [B, L, D]
        """
        d1 = data.size(0)  # batch size
        d3 = data.size(2)  # 特征维度

        # 预处理：对sample维度求和平均
        x = data.unsqueeze(-2)
        x = data.view(d1, -1, self.sample, d3)
        x = torch.sum(x, dim=-2).squeeze(-2)
        x = torch.div(x, self.sample)

        # 位置编码
        x = self.pos_encoding(x)
        # 水平Transformer
        x = self.transformer(x)

        if self.use_vertical:
            # 垂直分支处理
            y = data.view(-1, self.maxlen, self.embed_dim//30, 30)
            y = torch.sum(y, dim=-2).squeeze(-2)
            y = y.transpose(-1, -2)
            if self.v_transformer is not None:
                y = self.v_transformer(y)
            # 聚合水平和垂直特征
            re = self._aggregate(x, y)
            predict = self.softmax(self.dense(re))
        else:
            # 只聚合水平特征
            re = self._aggregate(x)
            predict = self.softmax(self.dense(re))

        # 返回分类结果
        # return self.feature_proj(re)
        return predict


class TwoStreamModel(nn.Module):
    def __init__(self, hlayers, vlayers, hheads, vheads, K, sample, num_class, maxlen, embed_dim):
        super(TwoStreamModel, self).__init__()
        self.sample = sample
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.transformer = MCAT(embed_dim, hlayers, hheads, maxlen//2)
        
        # 位置编码
        # self.pos_encoding = GRE(embed_dim, maxlen//2, K)
        # self.pos_encoding_v = GRE(self.maxlen, 30, K)

        # 可学习位置编码
        self.pos_encoding = LearnablePositionalEncoding(embed_dim, maxlen//2)
        self.pos_encoding_v = LearnablePositionalEncoding(self.maxlen, 30)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.use_vertical = vlayers > 0

        self.kernel_num = 128
        self.kernel_num_v = 16
        self.filter_sizes = [20, 40]
        self.filter_sizes_v = [2, 4]

        if self.use_vertical:
            self.v_transformer = Transformer(self.maxlen, vlayers, vheads, use_relative=True, max_len=self.maxlen)
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
            if self.v_transformer is not None:
                y = self.v_transformer(y)
            re = self._aggregate(x, y)
            predict = self.softmax(self.dense(re))
        else:
            re = self._aggregate(x)
            predict = self.softmax(self.dense(re))

        # return self.feature_proj(re)
        return predict

    # 新增：用于提取256维特征的方法
    def extract_feature(self, data):
        d1 = data.size(0)
        d3 = data.size(2)
        x = data.unsqueeze(-2)
        x = data.view(d1, -1, self.sample, d3)
        x = torch.sum(x, dim=-2).squeeze(-2)
        x = torch.div(x, self.sample)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        if self.use_vertical and self.v_transformer is not None:
            y = data.view(-1, self.maxlen, self.embed_dim//30, 30)
            y = torch.sum(y, dim=-2).squeeze(-2)
            y = y.transpose(-1, -2)
            y = self.v_transformer(y)
            re = self._aggregate(x, y)
        else:
            re = self._aggregate(x)
        return self.feature_proj(re)

