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
nat_2011 = ['PRECIP_LN', 'TRANGE_MED', 'GDD_MED', 'ELEV_MED', 'SLOPE_CBRT', 'ERODIBLE_P', 'LOWGRAD_CB', 'CAT_BFI']

dist_2011 = ['URBAN_LOG', 'AG_LOG', 'IMPERV_LOG', 'ROAD_LOG']

# gdf_agg = gdf.dissolve(by='HUC_12', aggfunc={'mean')

## Plot
df = gdf[disturbance]
