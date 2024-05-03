from scipy.spatial import distance_matrix
import scipy.sparse as sp
import pandas as pd
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch
import random
from torch_geometric.utils import dropout_adj, convert
from setup import seed

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh*max_sim)[0]
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    # print('building edge relationship complete')
    idx_map =  np.array(idx_map)
    
    return idx_map


def build_relationship_knn(x, k):
    # Compute the KNN graph
    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nn.fit(x)
    distances, indices = nn.kneighbors(x)

    # Convert indices to edges
    idx_map = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:
            idx_map.append([i, neighbor])

    return np.array(idx_map)

def load_heloc(dataset, predict_attr, path):

    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    idx_features_labels.replace({'RiskPerformance' : { 'Bad' : 1, 'Good' : 0}}, inplace=True)
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(idx_features_labels[header].values)

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship_knn(features_scaled, k=6) #thresh=0.8
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(features_scaled, dtype=np.float32) #without RiskPerformance
    labels = idx_features_labels[predict_attr].values 

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    # Calculate number of samples for each label in test and train sets
    num_label_1_test = int(label_idx_1.size * 0.3)
    num_label_0_test = int(label_idx_0.size * 0.3)
    num_label_1_train = label_idx_1.size - num_label_1_test
    num_label_0_train = label_idx_0.size - num_label_0_test

    # Split label indices for label 1 and label 0 into test and train sets
    idx_train_1 = label_idx_1[:num_label_1_train]
    idx_test_1 = label_idx_1[num_label_1_train:num_label_1_train+num_label_1_test]
    idx_train_0 = label_idx_0[:num_label_0_train]
    idx_test_0 = label_idx_0[num_label_0_train:num_label_0_train+num_label_0_test]

    # Concatenate indices to get final train and test sets
    idx_train = np.concatenate((idx_train_0, idx_train_1))
    idx_test = np.concatenate((idx_test_0, idx_test_1))

    # Shuffle final train and test sets
    random.shuffle(idx_train)
    random.shuffle(idx_test)

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
   
    return adj, features, labels, idx_train, idx_test 