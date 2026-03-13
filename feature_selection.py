#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:31:45 2026

@author: dawn.urycki
"""

import geopandas as gpd
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.datasets import load_iris

class FeatureSelectorUnsupervised:
    def __init__(self, dataframe):
        self.df = dataframe

    def tree_based_feature_importance(self, n_select_features=0.5):
        """
        Performs unsupervised feature selection by training a RandomForest to 
        discriminate between real and synthetic data, then returning the most 
        important features based on the classifier's feature importance scores.
        
        n_select_features: The number of features to select (int) or 
                           proportion (float, 0 to 1) of features to keep.
        """
        n_samples, n_features = self.df.shape
        
        # 1. Generate synthetic data (Addcl2 method: uniform sampling over bounds)
        #    This method is effective for identifying structure in the real data.
        synthetic_data = pd.DataFrame(np.random.uniform(
            low=self.df.min().values, 
            high=self.df.max().values, 
            size=self.df.shape
        ), columns=self.df.columns)
        
        scaler = MinMaxScaler()
        
        # 2. Combine real and synthetic data and create labels
        combined_df = pd.concat([self.df, synthetic_data], axis=0)
        
        # scaler = MinMaxScaler()
        # scaled_df = scaler.fit_transform(combined_df)
        
        # Labels: 1 for real data, 0 for synthetic data
        labels = np.array([1] * n_samples + [0] * n_samples)
        
        # 3. Train a RandomForestClassifier to discriminate
        #    The feature importances reflect which features best differentiate 
        #    real from synthetic data, indicating features with consistent structure.
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_clf.fit(combined_df, labels)
        
        # 4. Get feature importances and rank them
        importances = rf_clf.feature_importances_
        feature_names = self.df.columns
        feature_importance_df = pd.DataFrame(
            {'feature': feature_names, 'importance': importances}
        ).sort_values(by='importance', ascending=False)
        
        # 5. Select top features
        if isinstance(n_select_features, float):
            num_selected = int(n_features * n_select_features)
        else:
            num_selected = n_select_features
            
        selected_features_names = feature_importance_df.head(num_selected)['feature'].tolist()
        
        print(f"\nTotal features: {n_features}")
        print(f"Selected features ({num_selected}): {selected_features_names}")
        
        return self.df[selected_features_names], feature_importance_df

# --- Example Usage ---
if __name__ == "__main__":
    
    data_dir = 'DATA'

    gdf = gpd.read_file(os.path.join(data_dir, 'access_by_huc6_us_v11' ,'access_by_huc6_us_v11.dbf'))
    attributes = pd.read_csv(os.path.join(data_dir, 'huc12attributes.csv'), skiprows=[1])
    cols = list(gdf.columns)
    huc_12 = gdf.HUC_12.unique()
    duplicate_huc_12 = gdf[gdf.duplicated(subset=['HUC_12'])].HUC_12.unique()

    immutable_scase = list(attributes.loc[attributes['Imm_or_dist']=='I'].Alias)
    immutable = [i.upper() for i in immutable_scase]
    disturbance_scase = list(attributes.loc[attributes['Imm_or_dist']=='D'].Alias)
    disturbance = [i.upper() for i in disturbance_scase]
    nat_2011 = ['PRECIP_LN', 'TRANGE_MED', 'GDD_MED', 'ELEV_MED', 'SLOPE_CBRT', 'ERODIBLE_P', 'LOWGRAD_CB']
    dist_2011 = ['URBAN_LOG', 'AG_LOG', 'IMPERV_LOG', 'ROAD_LOG']
    
    
    
    df = gdf[immutable]
    
    # Instantiate the FeatureSelectorUnsupervised class
    selector = FeatureSelectorUnsupervised(df)
    
    # Perform feature selection, keeping top 20 features
    selected_features_df, feature_importances = selector.tree_based_feature_importance(n_select_features=20)
    
    print("\nFeature Importances:")
    print(feature_importances.head(20))

    # The selected_features_df can now be used for clustering algorithms (e.g., K-Means, hierarchical clustering)
    # The clustering results using the reduced set of features might be more accurate.

