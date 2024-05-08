from sklearn.metrics import f1_score, roc_auc_score
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from .train import train
from .evaluate import evaluate

def train_and_evaluate(model, optimizer, features, edge_index, labels, idx_train, idx_test, device, num_epochs, plot_loss=False, save_plot=False, tuning=False):
    train_losses = []
    test_losses = []
    best_auc = float('-inf')
    
    # Convert idx_train and idx_test to tensor if they are not
    if isinstance(idx_train, np.ndarray):
        idx_train = torch.tensor(idx_train, dtype=torch.long, device=device)
    if isinstance(idx_test, np.ndarray):
        idx_test = torch.tensor(idx_test, dtype=torch.long, device=device)
    
    # Ensure labels are a tensor on the correct device
    labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    
    epoch_iterator = tqdm(range(num_epochs + 1), desc='Epochs', unit='epoch')
    
    for epoch in epoch_iterator:
        loss_train = train(model, optimizer, features, edge_index, labels, idx_train, device)
        loss_test, auc_roc_val, f1_val = evaluate(model, features, edge_index, labels, idx_test, device)
        
        train_losses.append(loss_train)
        test_losses.append(loss_test)
        
        # Save the model if the current AUC is the best
        if auc_roc_val > best_auc:
            best_auc = auc_roc_val
            save_path = f'./models/weights/{args.arch}_weights.pth'
            config = {
                'nhid': model.nhid,
                'nfeat': model.nfeat,
                'dropout': model.dropout
            }
            if args.arch == 'GAT':
                config['num_heads'] = model.num_heads
                config['num_layers'] = model.num_layers

            torch.save({
                'state_dict': model.state_dict(),
                'config': config,
                'train_losses': train_losses,
                'test_losses': test_losses
            }, save_path)

        # Update progress bar with loss information
        if epoch % 100 == 0:
            epoch_iterator.set_postfix({'Train Loss': f"{loss_train:.4f}", 'Val Loss': f"{loss_test:.4f}", 'Val AUC': f"{auc_roc_val:.4f}"})
    
    model.eval()
    output = model(features, edge_index)
    probs = torch.sigmoid(output)
    probs_gnn_1 = probs.squeeze()
    probs_gnn_array = np.vstack((1 - probs_gnn_1.detach().cpu().numpy(), probs_gnn_1.detach().cpu().numpy())).T
    preds = (output.squeeze() > 0).type_as(labels)
    f1_test = f1_score(labels.cpu().numpy()[idx_test.cpu().numpy()], preds[idx_test.cpu().numpy()])
    auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu().numpy()], probs_gnn_array[idx_test.cpu().numpy()][:, 1])


    if plot_loss:
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(test_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title("Loss over epochs")
        if save_plot:
            plt.savefig(f'GIN_loss_plot.png')
        plt.show()

    return auc_roc_test if tuning else (auc_roc_test, probs_gnn_array, f1_test)