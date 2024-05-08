
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

class GIN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GIN, self).__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(nfeat, nhid),
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
        )
        self.conv1 = GINConv(self.mlp1)
        self.fc = nn.Linear(nhid, 1)  # Assuming binary classification
        self.dropout = dropout  # Storing dropout rate
        self.nfeat = nfeat
        self.nhid = nhid

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)  # Using dropout functionally
        x = self.fc(x)
        return x