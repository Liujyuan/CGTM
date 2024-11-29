import math
import torch
from torch import nn, optim
import torch.nn.functional as F
def check_nan(tensor):
    if torch.isnan(tensor).any():
        print("Tensor contains NaN values.")
        tensor[torch.isnan(tensor)] = 0  # 替换为0或其他值
    return tensor

class GIA(nn.Module):
    def __init__(self, vocab,in_features,eps):
        super(GIA, self).__init__()

        self.gcn = nn.ModuleList()
       # self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.weight = nn.Parameter(torch.FloatTensor(in_features,vocab))  #V,H
        self.eps=eps
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)


    def forward(self, x, adj):
        for i in range(len(self.eps)):
            x = 1 * x + self.eps[i] * x @ adj
        return x
