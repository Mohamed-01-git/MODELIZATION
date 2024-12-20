#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:02:56 2024

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
            mean_score = corr.mean(dim=None ,skipna=True).to_array().values

            # Append period-specific data to the list
            center_data.append({
                "center": file_name,
                "metric": metric,
                "period": period,
                "mean_score": mean_score
            })

        # Create a dataframe for this center
        center_df = pd.DataFrame(center_data)

        # Convert columns to appropriate types
        center_df['mean_score'] = center_df['mean_score'].apply(lambda x: float(x[0]) if isinstance(x, np.ndarray) else float(x))
        center_df['period'] = center_df['period'].astype('category')  # Convert 'period' to categorical
        center_df['center'] = center_df['center'].astype(str)  # Ensure 'file_name' is string
        center_df['metric'] = center_df['metric'].astype(str)  # Ensure 'metric' is string

        # Append to the list of dataframes
        all_dataframes.append(center_df)

    # Combine dataframes for all centers
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    return combined_df

# Example usage
aggr = "1m"  # Aggregation type
metric = "corr"  # Metric type

# Generate the combined dataframe
rps_RR_df = create_combined_dataframe("1m", "rps")
rela_RR_df = create_combined_dataframe("1m", "rela")  
roc_RR_df = create_combined_dataframe("1m", "roc")
rocss_RR_df=  create_combined_dataframe("1m", "rocss")
bs_RR_df=  create_combined_dataframe("1m", "bs")

# Save DataFrames to CSV files
rps_RR_df.to_csv("/home/mohamed/EHTPIII/MODELISATION/output/rps_RR_df.csv", index=False)
rela_RR_df.to_csv("/home/mohamed/EHTPIII/MODELISATION/output/rela_RR_df.csv", index=False)
roc_RR_df.to_csv("/home/mohamed/EHTPIII/MODELISATION/output/roc_RR_df.csv", index=False)
rocss_RR_df.to_csv("/home/mohamed/EHTPIII/MODELISATION/output/rocss_RR_df.csv", index=False)
bs_RR_df.to_csv("/home/mohamed/EHTPIII/MODELISATION/output/bs_RR_df.csv", index=False)

file_names = ['rps_RR_df','roc_RR_df','rocss_RR_df']

# for  file_name in file_names :
    
#     df=pd.read_csv(f'/home/mohamed/EHTPIII/MODELISATION/output/{file_name}.csv')

#     g = sns.catplot(data=df, x="center", y="mean_score", col="period", hue="period", kind="bar",palette="ch:start=.2,rot=-.3")

#     g.set_axis_labels("Center", file_name.split("_")[0])

#     g.fig.suptitle(f'{file_name.split("_")[0].upper()} : RR', y=0.9, fontsize=16)    
    

#     g.set_xticklabels(rotation=90)

#     g.fig.tight_layout() 

#     plot_path = f"/home/mohamed/EHTPIII/MODELISATION/output/plots/{file_name.split('_')[0]}_RR_cat.png"
#     plt.savefig(plot_path, dpi=400)
    
#     plt.show()



# List of dataframes and corresponding file names
# dataframes = [corr_df, rmse_df, rsquared_df, corr_pval_df,bs_df, rela_df, roc_df, rocss_df]
# file_names = ['rps_RR_df','rela_RR_df','roc_RR_df','rocss_RR_df','bs_RR_df']

for  file_name in file_names :
    
    df=pd.read_csv(f'/home/mohamed/EHTPIII/MODELISATION/output/{file_name}.csv')

    g = sns.catplot(data=df, x="center", y="mean_score", col="period", hue="mean_score", kind="bar",palette="ch:start=.2,rot=-.3")

    g.set_axis_labels("Center", file_name.split("_")[0])

    g.fig.suptitle(f'{file_name.split("_")[0].upper()} : RR', y=0.9, fontsize=16)    
    

    g.set_xticklabels(rotation=90)

    g.fig.tight_layout() 

    plot_path = f"/home/mohamed/EHTPIII/MODELISATION/output/plots/{file_name.split('_')[0]}_RR_all.png"
    plt.savefig(plot_path, dpi=400)
    
    plt.show()
