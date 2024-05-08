import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, num_heads=3, num_layers=1):
        super(GAT, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.num_heads = num_heads
        
        # Input layer
        self.gat_layers.append(GATConv(nfeat, nhid, heads=num_heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(1, num_layers):
            self.gat_layers.append(GATConv(nhid * num_heads, nhid, heads=num_heads, dropout=dropout))
        
        # Output layer
        self.gat_layers.append(GATConv(nhid * num_heads, 1, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.gat_layers[:-1]):
            x = layer(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat_layers[-1](x, edge_index)
        return x