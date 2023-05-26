import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, num_heads=3, num_layers=1):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()

        # Input layer
        self.gat_layers.append(GATConv(nfeat, nhid, heads=num_heads, dropout=dropout))

        # Hidden layers
        for _ in range(1, num_layers):
            self.gat_layers.append(GATConv(nhid * num_heads, nhid, heads=num_heads, dropout=dropout))

        # Output layer
        self.gat_layers.append(GATConv(nhid * num_heads, nclass, heads=num_heads, dropout=dropout, concat=False))

    def forward(self, x, edge_index):
        h = x

        # Propagate through each GAT layer
        for layer in range(self.num_layers):
            h = self.gat_layers[layer](h, edge_index).flatten(1)
            h = F.elu(h)

        # Output layer
        x = self.gat_layers[-1](h, edge_index)

        return x