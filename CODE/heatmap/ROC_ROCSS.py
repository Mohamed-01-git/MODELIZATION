#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:35:59 2024

@author: mohamed
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import seaborn as sns
import regionmask
import calendar

SCOREDIR ="/home/mohamed/EHTPIII/MODELISATION/DATA/DATASET/OUT/SF/scores"
# config = dict(list_vars=['tprate'], hcstarty=1993, hcendy=2016, start_month=11)
details = "_1993-2016_monthly_mean_5_234_45_-30_-2.5_60"
available_files = ["ukmo_602", "meteo_france_8", "ecmwf_51", "eccc_3", "eccc_2", "dwd_21", "cmcc_35"]
VARNAMES = {'tprate': 'RR'}
metrics = ["rmse", "corr","rsquared"]

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
    file_link_t2m = f'{SCOREDIR}/{file_name}_1993-2016_monthly_mean_{mois}_234_45_-30_-2.5_60_{period}.{aggr}.{metric}.nc'
    data_t2m= xr.open_dataset(file_link_t2m)
    data_t2m = data_t2m.assign_coords(lon=(((data_t2m.lon + 180) % 360) - 180)).sortby('lon')
    
    file_link_RR = f'{SCOREDIR}/{file_name}_1993-2016_monthly_mean_{mois}_234_45_-30_-2.5_60_{period}.{aggr}.RR.{metric}.nc'
    data_RR= xr.open_dataset(file_link_RR)
    data_RR = data_RR.assign_coords(lon=(((data_RR.lon + 180) % 360) - 180)).sortby('lon')
    return data_t2m, data_RR

# for file in available_files:
#     df1,df2=load_data(file,"1m","corr","djf")
#     print(df1.dims)


def get_mask(zone,df_t2m,df_rr):
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
    mask=countries.mask(df_t2m)
    mask=mask.transpose("lat","lon")
    if zone == "NorthAfrica":
        NA_country_names = ['Algeria','Egypt','Libya','Mauritania','Morocco','Tunisia']
    elif zone == "ArabianPeninsula":
        NA_country_names = ['Saudi Arabia','Yemen','Oman',
                            'United Arab Emirates','Kuwait',
                            'Qatar','Syria','Iraq',
                            'Jordan']
    
    na_indices =[countries.map_keys(name) for name in NA_country_names]
    broadcasted_mask = np.broadcast_to(mask.data, (df_t2m.sizes['forecastMonth'], *mask.shape))
    DATA_t2m=df_t2m.where(np.isin(broadcasted_mask, na_indices))
    mask=countries.mask(df_rr)
    mask=mask.transpose("lat","lon")
    broadcasted_mask_2 = np.broadcast_to(mask.data, (df_rr.sizes['forecastMonth'], *mask.shape))
    DATA_rr=df_rr.where(np.isin(broadcasted_mask_2, na_indices))
    return DATA_t2m , DATA_rr


def create_combined_dataframe(aggr, metric,mask_it,zone):
    all_dataframes = []

    for file_name in available_files:
        # List to hold data for all periods
        center_data = []

        for period in periods:
            # Load data for the current period
            data_t2m,data_RR = load_data(file_name, aggr, metric, period)
            if mask_it==True:
                data_t2m,data_RR = get_mask(zone,data_t2m,data_RR)
            
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


roc_df = create_combined_dataframe("1m", "roc",False,"mena")
rocss_df = create_combined_dataframe("1m", "rocss",False,"mena")  
bs_df = create_combined_dataframe("1m", "bs",False,"mena") 

roc_df_NorthAfrica = create_combined_dataframe("1m", "roc",True,"NorthAfrica")
rocss_df_NorthAfrica = create_combined_dataframe("1m", "rocss",True,"NorthAfrica")  
bs_df_NorthAfrica = create_combined_dataframe("1m", "bs",True,"NorthAfrica")

roc_df_ArabianPeninsula = create_combined_dataframe("1m", "roc",True,"ArabianPeninsula")
rocss_df_ArabianPeninsula = create_combined_dataframe("1m", "rocss",True,"ArabianPeninsula")  
bs_df_ArabianPeninsula = create_combined_dataframe("1m", "bs",True,"ArabianPeninsula")

TYPE=[ "lead_time", "category"]
variable=["T2M","RR"]
def plot_roc(df,variable,TYPE,mask_it,zone):
    fig,axe=plt.subplots(nrows=3,ncols=3,figsize=(25, 18))
    axe=axe.flatten()
    
    centers=df.center.unique()
    for i, center in enumerate(centers):
        df_center = df[df['center'] == center]
        df_temp=df_center.pivot(index= TYPE, columns="period", values=f"mean_{variable}_{TYPE}")
        # df_temp.columns= [calendar.month_abbr[m] for m in df_temp.columns]
        sns.heatmap(df_temp,  fmt=".2f", cmap="Blues", 
                    ax=axe[i],annot=True,annot_kws={"size": 22},
                    vmin=np.nanmin(df[f"mean_{variable}_{TYPE}"].values),
                    vmax=np.nanmax(df[f"mean_{variable}_{TYPE}"].values),
                    cbar_kws={"shrink": 1})
        
        cbar = axe[i].collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        axe[i].set_xlabel("",fontsize=15)
        axe[i].set_ylabel(f"{TYPE}",fontsize=20)
        axe[i].set_title(f'{center}',fontsize=20)
        sns.set(font_scale=2)

    subtitle=f"{df.metric[0]}  for {variable}  per  {TYPE}  {zone}"

    fig.suptitle(subtitle, fontsize=20, fontweight='bold', y=0.981)  
    # fig.suptitle(f"{df.metric[0]} {variable} / {TYPE} North Africa", fontsize=16, fontweight='bold', y=0.981)
    for j in range(i + 1, len(axe)):
        fig.delaxes(axe[j])

    file_out=f"{df.metric[0]}_{variable}_{TYPE}_{zone}.png"

    plt.savefig(f'/home/mohamed/EHTPIII/MODELISATION/Report_25_11/plots/prob/{df.metric[0]}/{file_out}')
    # plt.savefig(f'/home/mohamed/EHTPIII/MODELISATION/REPORT/Report_25_11/plots/prob/{df.metric[0]}/{df.metric[0]}_{variable}_{TYPE}_North_Africa.png')
    # plt.savefig(f'/home/mohamed/EHTPIII/MODELISATION/REPORT/Report_25_11/plots/prob/{df.metric[0]}/{df.metric[0]}_{variable}_{TYPE}.png')
        
    plt.tight_layout()
    plt.show()       

# for mask_it,df in zip([False]*3+[True]*3,[roc_df,rocss_df,bs_df,roc_df_masked,rocss_df_masked,bs_df_masked] ):
#     for i in [0,1]:
#         plot_roc(df,"RR",TYPE[i],mask_it)
#         plot_roc(df,"T2M",TYPE[i],mask_it)

mena_files=[roc_df,rocss_df,bs_df]
northafrica_files=[roc_df_NorthAfrica,rocss_df_NorthAfrica,bs_df_NorthAfrica]
arabian_peninsula_files=[roc_df_ArabianPeninsula,rocss_df_ArabianPeninsula,bs_df_ArabianPeninsula]


for df in mena_files : 
    for i in [0,1]:
        plot_roc(df,"RR",TYPE[i],False,"mena")
        plot_roc(df,"T2M",TYPE[i],False,"mena")

for df in northafrica_files : 
    for i in [0,1]:
        plot_roc(df,"RR",TYPE[i],False,"NorthAfrica")
        plot_roc(df,"T2M",TYPE[i],False,"NorthAfrica")
        
for df in arabian_peninsula_files : 
    for i in [0,1]:
        plot_roc(df,"RR",TYPE[i],False,"ArabianPeninsula")
        plot_roc(df,"T2M",TYPE[i],False,"ArabianPeninsula")