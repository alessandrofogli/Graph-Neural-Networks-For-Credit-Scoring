from sklearn.metrics import f1_score, roc_auc_score
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def evaluate(model, features, edge_index, labels, idx_test, device):
    model.eval()
    output = model(features, edge_index)
    
    preds = (output.squeeze() > 0).type_as(labels)
    loss_test = F.binary_cross_entropy_with_logits(output[idx_test], labels[idx_test].unsqueeze(1).float().to(device))
    
    auc_roc_val = roc_auc_score(labels.cpu().numpy()[idx_test], output.detach().cpu().numpy()[idx_test])
    f1_val = f1_score(labels[idx_test].cpu().numpy(), preds[idx_test].cpu().numpy())
    
    return loss_test.item(), auc_roc_val, f1_val