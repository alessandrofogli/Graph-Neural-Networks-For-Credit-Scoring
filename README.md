# Graph Neural Network for Credit Risk Classification

This repository contains the implementation of a graph neural network (GNN) model designed to predict credit risk using financial and non-financial data represented in a graph structure. The project compares traditional logistic regression model with advanced GNN models like GCN (Graph Convolutional Network) and GIN (Graph Isomorphism Network).

## Project Structure

The repository is organized as follows:

- `data/`: Contains raw and processed datasets.
- `models/`: Includes the model architectures (GCN, GIN, GAT), model weights and performance scores.
- `src/`: Source code for preprocessing, training, and utility functions.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and testing.

## Getting Started

### Prerequisites

Ensure you have Python 3.8 installed. It's recommended to use a virtual environment:

```bash
conda create -n myenv python=3.8
conda activate myenv
```

### Installation
Install the required dependencies:

```bash
pip install -r requirements.txt
```
## Available Models

The module supports the following GNN architectures:
- **GIN**: Graph Isomorphism Network
- **GCN**: Graph Convolutional Network
- **GAT**: Graph Attention Network

## Usage

### Command Line Arguments

The main script is designed to be run from the command line with the following arguments:

- `--mode`: Specifies the mode of operation. It can be either `train` or `retrieve`.
  - `train`: Trains a new model using the provided data and saves the optimization study results.
  - `retrieve`: Retrieves the best parameters from a previously saved optimization study and evaluates the model.
- `--arch`: Specifies the architecture of the GNN to use. Options are `GIN`, `GCN`, and `GAT`. Default is `GIN`.

### Examples

1. **Training a Model**:
```bash
python main.py --mode train --arch GAT
```

    │   

```
Exploring-the-potential-of-Graph-Neural-Networks-for-Credit-Scoring/
│
├── data/
│   ├── credit/
│   ├── FICO/
│   └── german/
│
├── models/
│   ├── __init__.py
│   ├── arch/
│   │   ├── __init__.py
│   │   ├── gcn.py
│   │   ├── gin.py
│   │   └── gat.py           
│   ├── saved_scores/
│   │       ├── __init__.py
│   │       └── study.pkl  # Artifact of the hyperparameter tuning trials, parameters and performance metrics
│   └── weights/
│           └── archname_WEIGHTS.pth  # Artifact of model's learned weights 
│
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuration settings for the project
│   ├── preprocessing/
│   │   ├── gnn/
│   │   │   └── graph_data_utils.py   # Preprocessing utilities
│   │   └── logistic_regression/
│   │       ├── correlation_feature_selector.py   # select uncorrelated features based on Gini
│   │       └── woe_processing.py   # WeightOfEvidence trasnformation
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── train_evaluate.py
│   │
│   └── visualization/
│       ├── __init__.py
│       └── plot_cap_curve.py
│
├── notebooks/
│   └── graph_visualization.ipynb.ipynb  # Jupyter notebook for exploration and testing
│
├── requirements.txt
├── .gitignore
├── README.md
├── setup.py          # Seeds setting
└── main.py           # Main script to run models
```
