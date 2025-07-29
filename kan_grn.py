import torch
import torch.nn as nn
import torch.nn.functional as F
from WaveletKAN import WaveletKANLinear

class KAN_GatesResidualNetwork(nn.Module):
    def __init__(self, units, dropout=0.2):
        super(KAN_GatesResidualNetwork, self).__init__()
        self.units = units
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(units)
        self.kan_dense_1 = WaveletKANLinear(units, units)
        self.kan_dense_2 = WaveletKANLinear(units, units)
        self.kan_gate_dense = WaveletKANLinear(units, units)

    def forward(self, x, context=None):
        x = self.elu(self.kan_dense_1(x))
        if context is not None:
            context = self.elu(self.kan_dense_2(context))
            x = x + context
        x = self.layer_norm(x)
        gate = self.sigmoid(self.kan_gate_dense(x))
        return gate 