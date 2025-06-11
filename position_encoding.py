
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def normal_pdf(pos, mu, sigma):
    a = pos - mu
    log_p = -1*torch.mul(a, a)/(2*sigma) - torch.log(sigma)/2
    return F.softmax(log_p, dim=1)

# Gaussian Position Endcoding (from THAT model)

class GRE(nn.Module):
    def __init__(self, d_model, total_size, K=10):
        super(GRE, self).__init__()
        self.embedding = nn.Parameter(torch.zeros([K, d_model], dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.register_buffer("positions", torch.arange(total_size).unsqueeze(1).repeat(1, K))  # 保证自动移动到设备

        interval = total_size / K
        mu_list = [torch.tensor(i * interval, dtype=torch.float32) for i in range(K)]
        self.mu = nn.Parameter(torch.stack(mu_list).unsqueeze(0))  # shape: [1, K]
        self.sigma = nn.Parameter(torch.ones(1, K) * 50.0)

    def forward(self, x):
        M = normal_pdf(self.positions.to(x.device), self.mu, self.sigma)
        pos_enc = torch.matmul(M, self.embedding)  # [total_size, d_model]
        pos_enc = pos_enc.unsqueeze(0)  # [1, total_size, d_model]
        return x + pos_enc  # 不要加 numpy！


class Gaussian_Position(nn.Module):
    def __init__(self, d_model, total_size, K=10):
        super(Gaussian_Position, self).__init__()
        #self.embedding = get_pe(d_model, K).to('cuda')
        #self.register_buffer('pe', self.embedding)
        self.embedding = nn.Parameter(torch.zeros([K, d_model], dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.positions = torch.tensor([i for i in range(total_size)], requires_grad=False).unsqueeze(1).repeat(1, K).to('cuda')
        s = 0.0
        interval = total_size / K
        mu = []
        for _ in range(K):
            mu.append(nn.Parameter(torch.tensor(s, dtype=torch.float), requires_grad=True))
            s = s + interval
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float).unsqueeze(0), requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor([torch.tensor([50.0], dtype=torch.float, requires_grad=True) for _ in range(K)]).unsqueeze(0))

    def forward(self, x):
        M = normal_pdf(self.positions, self.mu, self.sigma)
        pos_enc = torch.matmul(M, self.embedding)
        #print(M)
        return x + pos_enc.unsqueeze(0).repeat(x.size(0), 1, 1)