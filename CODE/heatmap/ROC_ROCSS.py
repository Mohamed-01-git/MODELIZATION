#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:53:10 2024

@author: mohamed
"""

import pandas as pd
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import calendar

SCOREDIR ="/home/mohamed/EHTPIII/MODELISATION/output/all_scores"
# config = dict(list_vars=['tprate'], hcstarty=1993, hcendy=2016, start_month=11)
details = "_1993-2016_monthly_mean_5_234_45_-30_-2.5_60"
available_files = ["ukmo_602", "meteo_france_8", "ecmwf_51", "eccc_3", "eccc_2", "dwd_21", "cmcc_35"]
VARNAMES = {'tprate': 'RR'}
metrics = ["roc", "rocss"]

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
    # file_link = f'{SCOREDIR}/{file_name}_1993-2016_monthly_mean_{mois}_234_45_-30_-2.5_60_{period}.{aggr}.RR.{metric}.nc'
    file_link_t2m = f'{SCOREDIR}/{file_name}_1993-2016_monthly_mean_{mois}_234_45_-30_-2_5_60_{period}_{aggr}_{metric}.nc'
    data_t2m= xr.open_dataset(file_link_t2m)
    data_t2m = data_t2m.assign_coords(lon=(((data_t2m.lon + 180) % 360) - 180)).sortby('lon')
    
    file_link_RR = f'{SCOREDIR}/{file_name}_1993-2016_monthly_mean_{mois}_234_45_-30_-2_5_60_{period}_{aggr}_RR_{metric}.nc'
    data_RR= xr.open_dataset(file_link_RR)
    data_RR = data_RR.assign_coords(lon=(((data_RR.lon + 180) % 360) - 180)).sortby('lon')
    return data_t2m, data_RR


def create_combined_dataframe(aggr, metric):
    all_dataframes = []

    for file_name in available_files:
        # List to hold data for all periods
        center_data = []

        for period in periods:
            # Load data for the current period
            data_t2m,data_RR = load_data(file_name, aggr, metric, period)

            # Compute the mean across all dimensions
            mean_score_RR_lead_time= data_RR.mean(dim=["lon", "lat",  "category"], skipna=True).to_array().values
            mean_score_RR_category= data_t2m.mean(dim=["lon", "lat","forecastMonth"], skipna=True).to_array().values
            
            mean_score_T2M_lead_time= data_RR.mean(dim=["lon", "lat",  "category"], skipna=True).to_array().values
            mean_score_T2M_category= data_t2m.mean(dim=["lon", "lat","forecastMonth"], skipna=True).to_array().values


            # mean_score = corr.mean(dim=["lon", "lat"], skipna=True).to_array().values
            mean_score_RR_lead_time = mean_score_RR_lead_time.flatten()
            mean_score_RR_category = mean_score_RR_category.flatten()
            mean_score_T2M_lead_time = mean_score_T2M_lead_time.flatten()
            mean_score_T2M_category = mean_score_T2M_category.flatten()
            

            # Append period-specific data to the list
            center_df=pd.DataFrame({
                "center": [file_name]*3,
                "metric": [metric]*3,
                "lead_time":[1,2,3],
                "period": [period]*3,
                "mean_RR_lead_time": mean_score_RR_lead_time,
                "mean_RR_category": mean_score_RR_category,
                "mean_T2M_lead_time": mean_score_T2M_lead_time,
                "mean_T2M_category": mean_score_T2M_category,
                "category": ["lower","middle","upper"]
            })

            all_dataframes.append(center_df)

    # Combine dataframes for all centers
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    return combined_df


roc_df = create_combined_dataframe("1m", "roc")
rocss_df = create_combined_dataframe("1m", "rocss")  
bs_df = create_combined_dataframe("1m", "bs") 


TYPE=[ "lead_time", "category"]
variable=["T2M","RR"]
def plot_roc(df,variable,TYPE):
    fig,axe=plt.subplots(nrows=3,ncols=3,figsize=(30,15))
    axe=axe.flatten()
    
    centers=df.center.unique()
    for i, center in enumerate(centers):
        df_center = df[df['center'] == center]
        df_temp=df_center.pivot(index= TYPE, columns="period", values=f"mean_{variable}_{TYPE}")
        # df_temp.columns= [calendar.month_abbr[m] for m in df_temp.columns]
        sns.heatmap(df_temp, annot=None, fmt=".2f", cmap="seismic", ax=axe[i])
        axe[i].set_xlabel("start_months")
        axe[i].set_ylabel(f"{TYPE}")
        axe[i].set_title(f'Center: {center}')
    fig.suptitle(f"{df.metric[0]} {variable} / {TYPE}", fontsize=16, fontweight='bold', y=0.981)  
    for j in range(i + 1, len(axe)):
        fig.delaxes(axe[j])
    plt.savefig(f'/home/mohamed/EHTPIII/MODELISATION/output/{df.metric[0]}_{variable}_{TYPE}.png')
        
    plt.tight_layout()
    plt.show()       

for x in [roc_df,rocss_df,bs_df] :
    for i in [0,1]:
        plot_roc(x,"RR",TYPE[i])
