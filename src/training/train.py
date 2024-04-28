from sklearn.metrics import f1_score, roc_auc_score
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def train(model, optimizer, features, edge_index, labels, idx_train, device):
    model.train()
    optimizer.zero_grad()
    output = model(features, edge_index)
    
    preds = (output.squeeze() > 0).type_as(labels)
    loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float().to(device))
    
    auc_roc_train = roc_auc_score(labels.cpu().numpy()[idx_train], output.detach().cpu().numpy()[idx_train])
    loss_train.backward()
    optimizer.step()
    
    return loss_train.item()