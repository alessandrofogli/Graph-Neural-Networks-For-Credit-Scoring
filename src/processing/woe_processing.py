import pandas as pd
from optbinning import BinningProcess, OptimalBinning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score


special_codes = [-9, -8, -7]

binning_fit_params = {
    "ExternalRiskEstimate": {"monotonic_trend": "descending"},
    "MSinceOldestTradeOpen": {"monotonic_trend": "descending"},
    "MSinceMostRecentTradeOpen": {"monotonic_trend": "descending"},
    "AverageMInFile": {"monotonic_trend": "descending"},
    "NumSatisfactoryTrades": {"monotonic_trend": "descending"},
    "NumTrades60Ever2DerogPubRec": {"monotonic_trend": "ascending"},
    "NumTrades90Ever2DerogPubRec": {"monotonic_trend": "ascending"},
    "PercentTradesNeverDelq": {"monotonic_trend": "descending"},
    "MSinceMostRecentDelq": {"monotonic_trend": "descending"},
    "NumTradesOpeninLast12M": {"monotonic_trend": "ascending"},
    "MSinceMostRecentInqexcl7days": {"monotonic_trend": "descending"},
    "NumInqLast6M": {"monotonic_trend": "ascending"},
    "NumInqLast6Mexcl7days": {"monotonic_trend": "ascending"},
    "NetFractionRevolvingBurden": {"monotonic_trend": "ascending"},
    "NetFractionInstallBurden": {"monotonic_trend": "ascending"},
    "NumBank2NatlTradesWHighUtilization": {"monotonic_trend": "ascending"}
}

def ob_univariate(x, y, feature_name, monotonic_trend, plot):

    """dev only"""

    # initialize binning
    optb = OptimalBinning(
        name=feature_name, 
        solver="mip", 
        monotonic_trend=monotonic_trend)

    # fit transform x_train
    x_woe = optb.fit_transform(x, y)

    if plot == True:
        binning_table = optb.binning_table
        binning_df = pd.DataFrame(binning_table.build())
        print(binning_df)
        print(binning_table.analysis(pvalue_test="chi2"))
        print(binning_table.plot(metric="event_rate",show_bin_labels=True))

    return x_woe

def create_woe_pipeline(X_train, X_test, y_train, y_test):

    binning_process = BinningProcess(variable_names = list(X_train.columns),
                                     special_codes=special_codes,
                                    binning_fit_params=binning_fit_params)

    logreg = LogisticRegression(fit_intercept=True, penalty='none')

    pipeline = Pipeline(steps=[("binning_process", binning_process), ("regressor", logreg)])

    pipeline.fit(X_train, y_train)
    
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred_f1 = (y_pred_proba >= 0.5).astype(int)  # Apply threshold and convert to binary labels
    

    y_pred = pipeline.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred_f1)
    roc_auc = roc_auc_score(y_test, y_pred)
    gini = roc_auc*2-1
    
    print(f"Gini : {gini}")
    print(f"ROC-AUC : {roc_auc}")
    print(f"F1 Score : {f1}")

    return pipeline