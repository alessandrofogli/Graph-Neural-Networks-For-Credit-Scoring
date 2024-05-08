import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.dropout = dropout
        self.body = GCN_Body(nfeat, nhid)
        self.fc1 = nn.Linear(nhid, nhid // 2)
        self.fc2 = nn.Linear(nhid // 2, 1)

    def forward(self, x, edge_index):
        x = self.body(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GCN_Body, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = self.gc2(x, edge_index)
        return F.relu(x)