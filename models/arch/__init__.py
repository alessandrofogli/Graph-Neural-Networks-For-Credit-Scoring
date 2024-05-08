# models/arch/__init__.py
from .gin import GIN
from .gcn import GCN
from .gat import GAT

__all__ = ['GIN', 'GCN', 'GAT']
