# Graph Neural Network for Credit Risk Classification

This repository contains the implementation of a graph neural network (GNN) model designed to predict credit risk using financial and non-financial data represented in a graph structure. The project compares traditional logistic regression model with advanced GNN models like GCN (Graph Convolutional Network) and GIN (Graph Isomorphism Network).

## Project Structure

The repository is organized as follows:

- `data/`: Contains raw and processed datasets.
- `models/`: Includes the model architectures (e.g., GCN, GIN).
- `output/`: Stores model weights and performance scores.
- `src/`: Source code for preprocessing, training, and utility functions.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and testing.
- `tests/`: Unit tests for the modules.
- `docs/`: Additional documentation for the project.

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Installation
Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Model
Navigate to the src/ directory and run:

```bash
python main.py
```

### Usage
To run the training process and evaluate the model, use the following command:

```bash
python src/main.py --model gcn --epochs 50 --batch-size 32
```


```
Exploring-the-potential-of-Graph-Neural-Networks-for-Credit-Scoring/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── __init__.py
│   ├── gnn/
│   │   ├── __init__.py
│   │   ├── gcn_model.py
│   │   ├── gin_model.py
│   │   └── ...             # Other GNN model scripts
│   └── logistic_regression/
│       ├── __init__.py
│       └── logistic_model.py
│
├── output/
│   ├── weights/
│   └── scores/
│
├── src/
│   ├── __init__.py
│   ├── main.py              # Main script to run models
│   ├── config.py            # Configuration settings for the project
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── preprocess.py   # Preprocessing utilities
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── train_evaluate.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_utils.py   # Utilities to handle and load graph data
│   └── visualization/
│       ├── __init__.py
│       └── plot_cap_curve.py
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Jupyter notebook for exploration and testing
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_training.py
│   └── test_models.py
│
├── docs/
│   └── project_documentation.md
│
├── requirements.txt
├── .gitignore
├── README.md
└── setup.py
```
