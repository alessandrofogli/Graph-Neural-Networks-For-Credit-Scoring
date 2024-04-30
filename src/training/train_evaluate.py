from sklearn.metrics import f1_score, roc_auc_score
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from .train import train
from .evaluate import evaluate

def train_and_evaluate(model, optimizer, features, edge_index, labels, idx_train, idx_test, device, num_epochs, architecture, plot_loss=False, save_plot=False, tuning=False):
    train_losses = []
    test_losses = []
    best_loss = float('inf')
    
    epoch_iterator = tqdm(range(num_epochs + 1), desc='Epochs')
    
    for epoch in epoch_iterator:
        t = time.time()
        loss_train = train(model, optimizer, features, edge_index, labels, idx_train, device)
        loss_test, auc_roc_val, f1_val = evaluate(model, features, edge_index, labels, idx_test, device)
        
        train_losses.append(loss_train)
        test_losses.append(loss_test)
        
        if epoch % 100 == 0:
            epoch_iterator.set_postfix({'Train Loss': loss_train, 'Val Loss': loss_test})
    
    model.eval()
    output = model(features, edge_index)
    probs = torch.sigmoid(output)
    probs_gnn_1 = probs.squeeze()

    # Ensuring CPU conversion happens on tensors
    probs_gnn_array = np.vstack((1 - probs_gnn_1.detach().cpu().numpy(), probs_gnn_1.detach().cpu().numpy())).T
    preds = (output.squeeze() > 0).type_as(labels)

    # Ensure labels and idx_test are tensors before using .cpu()
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    idx_test_np = idx_test.cpu().numpy() if isinstance(idx_test, torch.Tensor) else idx_test
    
    f1_test = f1_score(labels_np[idx_test_np], preds[idx_test_np])
    auc_roc_test = roc_auc_score(labels_np[idx_test_np], probs_gnn_array[idx_test_np][:, 1])

    if plot_loss:
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(test_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.title("Loss over epochs")
        if save_plot:
            plt.savefig('loss_plot.png', dpi=200)
        plt.show()

    return auc_roc_test if tuning else (auc_roc_test, probs_gnn_array, f1_test)
