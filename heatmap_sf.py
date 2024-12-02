#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 17:29:27 2024

@author: mohamed
"""

import pandas as pd
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt

SCOREDIR = "/home/mohamed/EHTPIII/MODELISATION/output/all_scores/scores_RR/kaggle/working/SF/scores"
config = dict(list_vars=['2m_temperature'], hcstarty=1993, hcendy=2016, start_month=11)
details = "_1993-2016_monthly_mean_5_234_45_-30_-2.5_60"
available_files = ["ukmo_602", "meteo_france_8", "ecmwf_51", "eccc_3", "eccc_2", "dwd_21", "cmcc_35"]
VARNAMES = {'t2m': '2m-T'}
metrics = ["corr", "rsquared", "rmse","corr_pval"]

period_to_month = {
    "djf": 11,
    "mam": 2,
    "jja": 5,
    "son": 8
    }
periods=["djf","mam","jja","son"]

def load_data(file_name, aggr, metric,period):
# Get the corresponding month
    mois = period_to_month.get(period)
    file_link = f'{SCOREDIR}/{file_name}_1993-2016_monthly_mean_{mois}_234_45_-30_-2.5_60_{period}.{aggr}.RR.{metric}.nc'
    corr = xr.open_dataset(file_link)
    corr = corr.assign_coords(lon=(((corr.lon + 180) % 360) - 180)).sortby('lon')
    return corr


def create_combined_dataframe(aggr, metric):
    all_dataframes = []

    for file_name in available_files:
        # List to hold data for all periods
        center_data = []

        for period in periods:
            # Load data for the current period
            corr = load_data(file_name, aggr, metric, period)

            # Compute the mean across all dimensions
            mean_score = corr.mean(dim=["lon","lat"],skipna=True).to_array().values
            mean_score = mean_score.flatten()

            # Append period-specific data to the list
            center_df=pd.DataFrame({
                "center": [file_name]*3,
                "metric": [metric]*3,
                "lead_time":[1,2,3],
                "period": [period]*3,
                "mean_score": mean_score,
                "start_month": [period_to_month[period]]*3
            })
        # Create a dataframe for this center
        # center_df = pd.DataFrame(center_data)

      

        # Append to the list of dataframes
            all_dataframes.append(center_df)

    # Combine dataframes for all centers
        combined_df = pd.concat(all_dataframes, ignore_index=True)

    return combined_df



# def create_combined_dataframe(aggr, metric):
#     all_dataframes = []

#     for file_name in available_files:
#         for period in periods:
#             # Load data for the current period
#             corr = load_data(file_name, aggr, metric, period)

#             # Compute the mean across all dimensions
#             mean_score = corr.mean(dim=["lon", "lat", "forecastMonth"], skipna=True).to_array().values

#             # Flatten the mean_score array (if it has extra dimensions)
#             if mean_score.ndim > 1:
#                 mean_score = mean_score.flatten()

#             # Ensure the mean_score has exactly 3 values
#             if len(mean_score) == 3:
#                 # Create a dataframe for this specific period and file
#                 center_data = pd.DataFrame({
#                     "center": [file_name] * 3,
#                     "metric": [metric] * 3,
#                     "period": [period] * 3,
#                     "tercile": ["lower", "middle", "upper"],
#                     "mean_score": mean_score
#                 })

#                 # Append to the list of dataframes
#                 all_dataframes.append(center_data)

#     # Combine dataframes for all centers and periods
#     combined_df = pd.concat(all_dataframes, ignore_index=True)
corr_RR_df = create_combined_dataframe("1m", "corr")
rsquared_RR_df = create_combined_dataframe("1m", "rsquared")  
rmse_RR_df = create_combined_dataframe("1m", "rmse")



corr_RR_df.to_csv("/home/mohamed/EHTPIII/MODELISATION/output/corr_RR2_df.csv", index=False)
rsquared_RR_df.to_csv("/home/mohamed/EHTPIII/MODELISATION/output/rsquared_RR2_df.csv", index=False)
rmse_RR_df.to_csv("/home/mohamed/EHTPIII/MODELISATION/output/rmse_RR2_df.csv", index=False)
