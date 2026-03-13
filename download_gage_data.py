#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:49:57 2026

@author: dawn.urycki
"""

import dataretrieval.nwis as nwis
import pandas as pd
import dataretrieval.nwis as nwis
import time
# Define parameters
# 'OR' is the state code for Oregon
# '00060' is the USGS parameter code for Discharge (streamflow) in cubic feet per second
# 'dv' indicates Daily Values
state_code = "OR"
parameter_code = "00060"
start_date = "1900-10-01"
end_date = "2025-09-30"
#gage_ids = ["14187500", "14191000", "14211010"]

# start_date = "2024-10-01"
# end_date = "2025-09-30"

start_time = time.perf_counter()


try:
    # 1. Fetch data for all sites in Oregon for the specified parameter and date range
    # get_dv returns a DataFrame and a metadata object
    df, metadata = nwis.get_dv(
        stateCd=state_code, 
#        sites = gage_ids,
        parameterCd=parameter_code, 
        start=start_date, 
        end=end_date
    )

    if not df.empty:
        # 2. Write to CSV
        #output_file = "oregon_stream_gage_data_202603.csv"
        output_file = "oregon_stream_gage_data_wy2025.csv"

        df.to_csv(output_file)
        print(f"Successfully saved {len(df)} records to {output_file}")
    else:
        print("No data found for the specified criteria.")

except Exception as e:
    print(f"An error occurred: {e}")

end_time = time.perf_counter()
elapsed_time = end_time - start_time #seconds
elapsed_hours = elapsed_time/60/60
print(f"Script executed in {elapsed_time:0.4f} seconds")

#%%
''''The following script uses the requests library to fetch CSV data 
directly from OWRD's API for a specific list of station IDs.'''

import requests
import io

# 1. Define OWRD Station IDs (Example: 14191000 is often OWRD/USGS shared)
# You can find these on the OWRD Hydrographics page
station_ids = ["14191000", "14321000"] 

start_date = "01/01/2024"
end_date = "01/31/2024"

all_data = []

print("Fetching data from OWRD...")

for station in station_ids:
    # OWRD uses a specific URL format for CSV downloads
    url = f"https://apps.wrd.state.or.us"
    params = {
        'station_nbr': station,
        'start_date': start_date,
        'end_date': end_date,
        'format': 'csv',
        'report_type': 'hourly' # Can be 'daily' or 'hourly'
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Read the CSV content into a DataFrame
        # OWRD often has header rows that need skipping or specific handling
        df = pd.read_csv(io.StringIO(response.text))
        
        if not df.empty:
            df['station_id'] = station  # Add ID column for tracking
            all_data.append(df)
            print(f"Successfully retrieved station {station}")
            
    except Exception as e:
        print(f"Failed to download station {station}: {e}")

# 2. Combine and save to CSV
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv("or_state_gages_data.csv", index=False)
    print(f"\nSaved all records to 'or_state_gages_data.csv'")
else:
    print("No data was retrieved.")
# %%time
# stationsToGet = stations_1a
# colnames = ['date', 'station', 'meanQ_cfs', 'notes'] 
# frames = []
# for station in stationsToGet:
#     frames.append((nwis.get_record(sites=str(station), service='dv', start=begin_date, parameterCd='00060')).reset_index())
# alldata = pd.concat(frames, ignore_index=True)

# %%time
# stationsToGet = stations_1a
# colnames = ['date', 'station', 'meanQ_cfs', 'notes'] 
# frames = []
# for station in stationsToGet:
#     frames.append((nwis.get_record(sites=str(station), service='dv', start=begin_date)).reset_index())
# alldata_1a = pd.concat(frames, ignore_index=True)