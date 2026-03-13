#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:24:42 2026

@author: dawn.urycki
"""

import geopandas as gpd
import pandas as pd
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from scipy import stats
from scipy.special import logit
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

data_dir = 'DATA'

gdf = gpd.read_file(os.path.join(data_dir, 'access_by_huc6_us_v11' ,'access_by_huc6_us_v11.dbf'))
attributes = pd.read_csv(os.path.join(data_dir, 'huc12attributes.csv'))
cols = list(gdf.columns)
huc_12 = gdf.HUC_12.unique()
duplicate_huc_12 = gdf[gdf.duplicated(subset=['HUC_12'])].HUC_12.unique()

immutable_scase = list(attributes.loc[attributes['Imm_or_dist']=='I'].Alias)
immutable = [i.upper() for i in immutable_scase]
disturbance_scase = list(attributes.loc[attributes['Imm_or_dist']=='D'].Alias)
disturbance = [i.upper() for i in disturbance_scase]

nat_2011 = ['AV_PPT_MM', 'AV_TEMPDIF', 'ME_DD50B', 'AV_ELEV', 'AV_SLOPE', 'ERODE_PER', 'LOWGRADKM']
nat_2011_tr = ['PRECIP_LN', 'TRANGE_MED', 'GDD_MED', 'ELEV_MED', 'SLOPE_CBRT', 'ERODIBLE_P', 'LOWGRAD_CB']
#nat_2011 = ['PRECIP_LN', 'TRANGE_MED', 'GDD_MED', 'ELEV_MED', 'SLOPE_CBRT', 'ERODIBLE_P', 'LOWGRAD_CB', 'CAT_BFI']

dist_2011 = ['URBAN_LOG', 'AG_LOG', 'IMPERV_LOG', 'ROAD_LOG']

scaler = MinMaxScaler()
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')


for i in nat_2011: gdf[i].plot(kind='hist', bins = 20, title=i); plt.show()

for i in nat_2011_tr: gdf[i].plot(kind='hist', bins = 20, title=i); plt.show()

for i, var in enumerate(nat_2011): 
    if var in ['AV_ELEV']:       
        scaled_df = pd.DataFrame(scaler.fit_transform(pd.DataFrame(gdf[var])))
        scaled_df = scaled_df*0.999 + 0.0005
        pd.DataFrame(logit(pd.DataFrame(scaled_df).dropna())[0]).plot(kind='hist', bins = 20, title=f'{var} (Logit)'); plt.show()
    if var in ['AV_SLOPE']:
        binned_data = kbd.fit_transform(gdf[var].dropna().values.reshape(-1, 1))
    else: 
        pd.DataFrame(stats.boxcox(gdf[var].add(1).dropna())[0]).plot(kind='hist', bins = 20, title=f'{var} (Box-Cox)'); plt.show()






df = gdf[nat_2011 + ['HUC_12', 'geometry']]
df_agg = df.dissolve(by='HUC_12', aggfunc='mean')
df_nonan = df_agg.dropna()[nat_2011]









divisive_model = BisectingKMeans(n_clusters=2, bisecting_strategy='biggest_inertia', random_state=42)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init='auto')

#labels = divisive_model.fit_predict(scaled_df)
kmeans.fit(scaled_df)
labels = kmeans.labels_

df_labelled = pd.DataFrame(scaled_df, columns = nat_2011, index = df_nonan.index)
df_labelled['label'] = labels
df_labelled['geometry'] = df_labelled.index.map(df_agg.geometry)
df_labelled = gpd.GeoDataFrame(df_labelled)

df_labelled.plot(column = 'label')
plt.show()

df_long = df_labelled.drop('geometry', axis =1).reset_index().melt(
    id_vars = ['HUC_12', 'label'], value_vars = nat_2011, var_name = 'measurement', value_name = '[]')


sns.boxplot(data=df_long, 
            x='measurement', 
            y='[]', 
            hue='label'
           )


'''
Combine duplicate HUC_12s (disconnected polygons)
# Assuming you have a GeoDataFrame called 'gdf'

# Example: Dissolve by a column named 'group_id'
# and sum the values in a column named 'population'
# while taking the first value of a column named 'name'

aggregated_gdf = gdf.dissolve(
    by='group_id',
    aggfunc={
        'population': 'sum',      # Sum the population values
        'name': 'first',          # Take the first name encountered
        'other_attr': np.mean     # Calculate the mean of another attribute
    }
)
'''