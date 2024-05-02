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


from models.gin import GIN
from src.processing.gnn.graph_data_utils import load_heloc
from src.training.train_evaluate import train_and_evaluate

def train_optimize_model(features, adj, labels, num_class, device, num_epochs=1500, n_splits=2):
    num_features = features.shape[1]
    edge_index = convert.from_scipy_sparse_matrix(adj)[0].to(device)
    features = features.to(device)
    labels = labels.to(device)

    kf = KFold(n_splits=n_splits, shuffle=True)
    study_name = "GIN_fico4_optimization_study"
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

        model = GIN(num_features, nhid, num_class, dropout).to(device)
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

def evaluate_saved_model(architecture, features, adj, labels, idx_test, num_class, best_params, device, plot_loss=False):
    print(f"Initializing the model with parameters: {best_params}")

    if not best_params:
        print("Best parameters not found.")
        return None

    try:
        model_path = f'./models/weights{architecture}_weights.pth'
        checkpoint = torch.load(model_path)
        model = GIN(nfeat=features.shape[1], nhid=best_params['nhid'], nclass=num_class, dropout=best_params['dropout']).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        print("Model loaded successfully.")
        
        if plot_loss:
            train_losses = checkpoint['train_losses']
            test_losses = checkpoint['test_losses']
            plt.plot(train_losses, label='Training Loss', color='blue')
            plt.plot(test_losses, label='Validation Loss', color='red')
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
    parser = argparse.ArgumentParser(description='Train or retrieve GIN model.')
    parser.add_argument('mode', choices=['train', 'retrieve'], help='Mode to execute')
    args = parser.parse_args()

    predict_attr = "RiskPerformance"
    path_heloc = "./data/FICO/"
    adj, features, labels, idx_train, idx_test = load_heloc('heloc', predict_attr, path=path_heloc)
    num_class = labels.unique().shape[0] - 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == 'train':
        best_params = train_optimize_model(features, adj, labels, num_class, device)
    if args.mode == 'retrieve':
        best_params = retrieve_model('GIN')
        if best_params:
            result = evaluate_saved_model('GIN', features, adj, labels, idx_test, num_class, best_params, device, plot_loss=True)
            if result:
                auc_roc, f1_score = result
                print(f'AUC-ROC: {auc_roc:.3f}, Gini: {(auc_roc*2-1):.3f}, F1-Score: {f1_score:.3f}')
            else:
                print("Evaluation failed due to model loading issues.")
        else:
            print("Failed to retrieve best parameters.")
