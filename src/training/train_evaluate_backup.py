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
    #for epoch in range(num_epochs + 1):
    for epoch in epoch_iterator:
        t = time.time()
        loss_train = train(model, optimizer, features, edge_index, labels, idx_train, device)
        loss_test, auc_roc_val, f1_val = evaluate(model, features, edge_index, labels, idx_test, device)
        
        train_losses.append(loss_train)
        test_losses.append(loss_test)
        '''
        if loss_test < best_loss:
            best_loss = loss_test
            if architecture == 'GIN':
                torch.save(model.state_dict(), './weights/GIN_weights.pt')
            elif architecture == 'GCN':
                torch.save(model.state_dict(), './weights/GCN_weights.pt')
            else:
                torch.save(model.state_dict(), './weights/GAT_weights.pt')
        '''
        if epoch % 100 == 0:
            epoch_iterator.set_postfix({'Train Loss': loss_train, 'Val Loss': loss_test})
    model.eval()
    output = model(features, edge_index)
    probs = torch.sigmoid(output)
    probs_gnn_1 = probs.squeeze()
    probs_gnn_array = np.vstack((1 - probs_gnn_1.cpu().detach().numpy(), probs_gnn_1.cpu().detach().numpy())).T
    preds = (output.squeeze() > 0).type_as(labels)
    f1_test = f1_score(labels[idx_test].cpu().numpy(), preds[idx_test].cpu().numpy())
    auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], probs_gnn_array[idx_test][:, 1])

    if plot_loss == True:
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(test_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)  # Add a grid
        plt.title("Loss over epochs")
        if save_plot==True:
            plt.savefig('loss_plot.png', dpi=200)  # Save the plot with 200 dpi resolution
        plt.show()

    if tuning == True:
        return auc_roc_test
    else:
        return auc_roc_test, probs_gnn_array, f1_test