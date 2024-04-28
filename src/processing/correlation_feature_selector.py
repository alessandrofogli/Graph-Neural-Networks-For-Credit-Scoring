import numpy as np

class CorrelationFeatureSelector:
    def __init__(self, features, corr_threshold=0.5, gini_threshold=0.08):
        self.features = features
        self.corr_threshold = corr_threshold
        self.gini_threshold = gini_threshold

    def select_features(self, df, ginis_df):
        X = df[self.features].copy()
        corr_mat = X.corr(method='spearman')

        np.fill_diagonal(corr_mat.values, 0)
        high_corr_mat = corr_mat[corr_mat.abs() > self.corr_threshold]
        high_corr_pair_list = sorted(set([tuple(sorted(pair)) for pair in list(high_corr_mat.stack().index)]))

        corr_features = []
        for tup in high_corr_pair_list:
            for item in tup:
                corr_features.append(item)
        flat_list = [item for couple in high_corr_pair_list for item in couple]
        ginis_df_corr = ginis_df[ginis_df['feature'].isin(flat_list)].sort_values(by='gini', ascending=False)
        selected_features = []
        dropped_features = []
        for feature_tuple in high_corr_pair_list:
            feature1, feature2 = feature_tuple
            gini1 = ginis_df_corr[ginis_df_corr['feature']==feature1]['gini'].values[0]
            gini2 = ginis_df_corr[ginis_df_corr['feature']==feature2]['gini'].values[0]
            if gini1 > gini2:
                selected_features.append(feature1)
                dropped_features.append(feature2)
            else:
                selected_features.append(feature2)
                dropped_features.append(feature1)
        ginis_df_uncorr = ginis_df[~ginis_df['feature'].isin(dropped_features)].sort_values(by='gini', ascending=False)
        mask = ginis_df_uncorr['gini'] >= self.gini_threshold
        best_variables = ginis_df_uncorr[mask]
        
        return best_variables