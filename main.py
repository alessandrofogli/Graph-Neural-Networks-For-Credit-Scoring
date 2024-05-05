import os
import pickle
import optuna
import torch
import torch.optim as optim
import numpy as np
from torch_geometric.utils import convert
import argparse
from sklearn.model_selection import KFold
import statistics
import matplotlib.pyplot as plt

from models.gin import GIN
from models.gcn import GCN  
from models.gat import GAT
from src.processing.gnn.graph_data_utils import load_heloc
from src.training.train_evaluate import train_and_evaluate
from src.training.evaluate import evaluate

def get_model(architecture, num_features, nhid, num_class, dropout, device, **kwargs):
    if architecture == 'GIN':
        return GIN(num_features, nhid, num_class, dropout).to(device)
    elif architecture == 'GCN':
        return GCN(num_features, nhid, num_class, dropout).to(device)
    elif architecture == 'GAT':
        num_heads = kwargs.get('num_heads', 8)  # Provide a default value if not specified
        num_layers = kwargs.get('num_layers', 1)  # Provide a default value if not specified
        return GAT(num_features, nhid, num_class, dropout, num_heads=num_heads, num_layers=num_layers).to(device)
    else:
        raise ValueError("Unsupported architecture specified")

def train_optimize_model(architecture, features, adj, labels, num_class, device, num_epochs=1500, n_splits=2):
    num_features = features.shape[1]
    edge_index = convert.from_scipy_sparse_matrix(adj)[0].to(device)
    features = features.to(device)
    labels = labels.to(device)

    kf = KFold(n_splits=n_splits, shuffle=True)
    study_name = f"{architecture}_fico_optimization_study"
    storage = f"./models/saved_scores/{study_name}.pkl"

    # Load or create a new study
    try:
        with open(storage, "rb") as f:
            study = pickle.load(f)
        print("Loaded previous study.")
    except FileNotFoundError:
        study = optuna.create_study(direction='maximize', study_name=study_name)
        print("Created new study.")


    # Define the objective function
    def objective(trial):
        dropout = trial.suggest_float('dropout', 0.2, 0.6)
        nhid = trial.suggest_int('nhid', 10, 60)
        lr = trial.suggest_float('lr', 1e-4, 1e-2)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2)

        model = get_model(architecture, num_features, nhid, num_class, dropout, device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        auc_roc_test = train_and_evaluate(model, optimizer, features, edge_index, labels, idx_train, idx_test, device, num_epochs, model.__class__.__name__, tuning=True)
        
        return auc_roc_test * 2 - 1  # Convert AUC to Gini

    # Perform optimization
    for fold, (train_idx, test_idx) in enumerate(kf.split(features)):
        idx_train, idx_test = train_idx, test_idx
        print(f"Optimizing on fold {fold+1}/{n_splits}...")
        study.optimize(objective, n_trials=2)

        # Optionally save the study after each fold
        with open(storage, "wb") as f:
            pickle.dump(study, f)

    print("Study saved.")
    best_trial = study.best_trial
    print('Best parameters:', best_trial.params)
    print('Best Gini:', best_trial.value)
    return best_trial.params

def retrieve_model(architecture):
    study_name = F"{architecture}_fico4_optimization_study"
    storage = f"./models/saved_scores/{study_name}.pkl"

    with open(storage, "rb") as f:
        study = pickle.load(f)

    best_params = study.best_params
    best_gini = study.best_value

    return best_params

def evaluate_saved_model(architecture, features, adj, labels, idx_test, num_class, device, plot_loss=False):
    print(f"Loading model weights for architecture: {architecture}")

    model_path = f'./models/weights/{architecture}_weights.pth'
    try:
        checkpoint = torch.load(model_path)
        config = checkpoint['config']
        # Initialize the model using the saved configuration
        model = GIN(nfeat=config['nfeat'], nhid=config['nhid'], nclass=config['nclass'], dropout=config['dropout']).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        print("Model loaded successfully.")

        if plot_loss:
            plt.plot(checkpoint['train_losses'], label='Training Loss', color='blue')
            plt.plot(checkpoint['test_losses'], label='Validation Loss', color='red')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title("Loss over epochs")
            plt.show()
    except Exception as e:
        print(f"Failed to load the model with error: {e}")
        return None

    model.eval()
    edge_index = convert.from_scipy_sparse_matrix(adj)[0].to(device)
    _, auc_roc_val, f1_val = evaluate(model, features, edge_index, labels, idx_test, device)
    
    return auc_roc_val, f1_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or retrieve GNN model.')
    parser.add_argument('--mode', choices=['train', 'retrieve'], help='Mode to execute')
    parser.add_argument('--arch', choices=['GIN', 'GCN', 'GAT'], help='Architecture to use')
    args = parser.parse_args()

    predict_attr = "RiskPerformance"
    path_heloc = "./data/FICO/"
    adj, features, labels, idx_train, idx_test = load_heloc('heloc', predict_attr, path=path_heloc)
    num_class = labels.unique().shape[0] - 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == 'train':
        best_params = train_optimize_model(args.arch, features, adj, labels, num_class, device)
    elif args.mode == 'retrieve':
        best_params = retrieve_model(args.arch)  # This function needs to be updated similarly to handle different architectures
        if best_params:
            result = evaluate_saved_model(
                architecture=args.arch,
                features=features,
                adj=adj,
                labels=labels,
                idx_test=idx_test,
                num_class=num_class,
                device=device,
                plot_loss=True
            )
            if result:
                auc_roc, f1_score = result
                print(f'AUC-ROC: {auc_roc:.3f}, Gini: {(auc_roc*2-1):.3f}, F1-Score: {f1_score:.3f}')
            else:
                print("Evaluation failed due to model loading issues.")
        else:
            print("Failed to retrieve best parameters.")