import torch
import torch.nn as nn
import torch.nn.functional as F

class GatesResidualNetwork(nn.Module):
    def __init__(self, units, dropout=0.2):
        super(GatesResidualNetwork, self).__init__()
        self.units = units
        self.dropout = nn.Dropout(dropout)

        self.dense_1 = nn.Linear(units, units)
        self.dense_2 = nn.Linear(units, units)
        self.elu = nn.ELU()
        self.gate_dense = nn.Linear(units, units)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(units)

    def forward(self, x, context=None):
        x = self.elu(self.dense_1(x))

        if context is not None:
            context = self.elu(self.dense_2(context))
            x = x + context

        x = self.layer_norm(x)
        gate = self.sigmoid(self.gate_dense(x))
        return gate
