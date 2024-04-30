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
    study_name = "GIN_fico3_optimization_study"
    storage = f"./models/saved_scores/{study_name}.pkl"

    # Try to load existing study, if not create new
    try:
        with open(storage, "rb") as f:
            study = pickle.load(f)
        print("Loaded previous study.")
    except FileNotFoundError:
        study = optuna.create_study(direction='maximize', study_name=study_name)

    for fold, (train_idx, test_idx) in enumerate(kf.split(features)):
        idx_train, idx_test = train_idx, test_idx

        def objective(trial):
            dropout = trial.suggest_float('dropout', 0.2, 0.6)
            nhid = trial.suggest_int('nhid', 10, 60)
            lr = trial.suggest_float('lr', 1e-4, 1e-2)
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2)

            model = GIN(num_features, nhid, num_class, dropout).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            auc_roc_test = train_and_evaluate(model, optimizer, features, edge_index, labels, idx_train, idx_test, device, num_epochs, model.__class__.__name__, tuning=True)
            return ((auc_roc_test * 2) - 1)

        study.optimize(objective, n_trials=3)

    # Save the entire study after all folds
    with open(storage, "wb") as f:
        pickle.dump(study, f)
    print("Study saved.")

    param_to_perf = {}
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            params_tuple = tuple(sorted(trial.params.items()))
            if params_tuple not in param_to_perf:
                param_to_perf[params_tuple] = []
            param_to_perf[params_tuple].append(trial.value)

    avg_performance = {params: statistics.mean(values) for params, values in param_to_perf.items()}
    best_params = max(avg_performance, key=avg_performance.get)

    best_params_dict = dict(best_params)
    print('Best Average Parameters:', best_params_dict)
    print('Best Average Gini:', avg_performance[best_params])

    return best_params_dict

def retrieve_model(architecture):
    study_name = F"{architecture}_fico3_optimization_study"
    storage = f"./models/saved_scores/{study_name}.pkl"

    with open(storage, "rb") as f:
        study = pickle.load(f)

    params = study.best_params
    best_gini = study.best_value

    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or retrieve GIN model.')
    parser.add_argument('mode', choices=['train', 'retrieve'], help='Mode to execute (train/retrieve)')
    args = parser.parse_args()

    predict_attr = "RiskPerformance"
    path_heloc = "./data/FICO/"
    adj, features, labels, idx_train, idx_test = load_heloc('heloc', predict_attr, path=path_heloc)
    num_class = labels.unique().shape[0] - 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == 'train':
        best_params = train_optimize_model(features, adj, labels, num_class, device)

    elif args.mode == 'retrieve':
        best_params = retrieve_model('GIN')

        model = GIN(nfeat=features.shape[1], nhid=best_params['nhid'], nclass=num_class, dropout=best_params['dropout']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
        num_epochs = 1500
        best_auc, gnn_probs, f1_sc = train_and_evaluate(model, optimizer, features, convert.from_scipy_sparse_matrix(adj)[0].to(device),
                                                        labels, idx_train, idx_test, device, num_epochs,
                                                        model.__class__.__name__, plot_loss=True, save_plot=True)
        print(f'Best parameters: {best_params}')
        print(f'AUC-ROC: {round(best_auc,3)}')
        print(f'Gini: {round(best_auc*2-1,3)}')
        print(f'F1-Score: {round(f1_sc,3)}')
