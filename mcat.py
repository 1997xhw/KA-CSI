import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class HAR_CNN(nn.Module):
    def __init__(self, d_model, d_ff, filters, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_ff, kernel_size=f, padding=f // 2),
                nn.BatchNorm1d(d_ff),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for f in filters
        ])

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, D, T]
        outs = [conv(x).unsqueeze(1) for conv in self.convs]
        out = torch.mean(torch.cat(outs, dim=1), dim=1)
        return out.permute(0, 2, 1)

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = nn.ModuleList([
            SublayerConnection(size, dropout),
            SublayerConnection(size, dropout)
        ])
        self.size = size

    def forward(self, x):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayers[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class MCAT(nn.Module):
    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5]):
        super().__init__()
        self.model = Encoder(
            EncoderLayer(hidden_dim, MultiHeadAttention(H, hidden_dim),
                         HAR_CNN(hidden_dim, hidden_dim, filters),
                         0.1),
            N
        )

    def forward(self, x):
        return self.model(x)
