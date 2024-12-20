#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 10:08:35 2024

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
        
        print(f"working on file {file_name} for {metric}")
        

        for period in periods:
            # Load data for the current period
            data_t2m,data_RR = load_data(file_name, aggr, metric, period)
            if mask_it==True:
                data_t2m,data_RR = get_mask(zone,data_t2m,data_RR)
            
            # data_t2m_masked,data_RR_masked=data_t2m,data_RR

            # Compute the mean across all dimensions
            mean_score_RR= data_RR.mean(dim=["lon", "lat"], skipna=True).to_array().values
            mean_score_T2M= data_t2m.mean(dim=["lon", "lat"], skipna=True).to_array().values


            # mean_score = corr.mean(dim=["lon", "lat"], skipna=True).to_array().values
            mean_score_RR = mean_score_RR.flatten()
            mean_score_T2M = mean_score_T2M.flatten()

            # Append period-specific data to the list
            center_df=pd.DataFrame({
                "center": [file_name]*3,
                "metric": [metric]*3,
                "lead_time":[1,2,3],
                "period": [period]*3,
                "mean_score_RR": mean_score_RR,
                "mean_scoArabianPeninsulare_T2M": mean_score_T2M,
            })

            all_dataframes.append(center_df)

    # Combine dataframes for all centers
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    return combined_df

#MENA
rmse_df= create_combined_dataframe("1m", "rmse",False,"mena")
corr_df= create_combined_dataframe("1m", "corr",False,"mena")  
acc_df= create_combined_dataframe("1m", "acc",False,"mena")
rsquared_df = create_combined_dataframe("1m", "rsquared",False,"mena")  
# NORTH AFRICA
rmse_df_NorthAfrica= create_combined_dataframe("1m", "rmse",True,"NorthAfrica")
corr_df_NorthAfrica= create_combined_dataframe("1m", "corr",True,"NorthAfrica") 
acc_df_NorthAfrica= create_combined_dataframe("1m", "acc",True,"NorthAfrica") 
rsquared_df_NorthAfrica= create_combined_dataframe("1m", "rsquared",True,"NorthAfrica")
# ARABIAN PENINSULA
rmse_df_ArabianPeninsula= create_combined_dataframe("1m", "rmse",True,"")
corr_df_ArabianPeninsula= create_combined_dataframe("1m", "corr",True,"ArabianPeninsula") 
acc_df_ArabianPeninsula= create_combined_dataframe("1m", "acc",True,"ArabianPeninsula") 
rsquared_df_ArabianPeninsula = create_combined_dataframe("1m", "rsquared",True,"ArabianPeninsula")
# rps_df=create_combined_dataframe("1m", "rps")


def plot_determinist(df,variable,mask_it,zone):
    fig,axe=plt.subplots(nrows=3,ncols=3,figsize=(25, 18))
    axe=axe.flatten()
    
    centers=df.center.unique()
    for i, center in enumerate(centers):
        df_center = df[df['center'] == center]
        df_temp=df_center.pivot(index="lead_time", columns="period", values=f"mean_score_{variable}")
        # df_temp.columns= [calendar.month_abbr[m] for m in df_temp.columns]
        sns.heatmap(df_temp,  fmt=".2f", cmap="Blues", 
                    ax=axe[i],annot=True,annot_kws={"size": 22},
                    vmin=np.nanmin(df[f"mean_score_{variable}"].values),
                    vmax=np.nanmax(df[f"mean_score_{variable}"].values),
                    cbar_kws={"shrink": 1})

        cbar = axe[i].collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        axe[i].set_xlabel("",fontsize=15)
        axe[i].set_ylabel("LEAD TIME",fontsize=20)
        axe[i].set_title(f'{center}',fontsize=20)

    subtitle=f"{df.metric[0]}  for {variable}  per  LEAD TIME {zone}"
    fig.suptitle(subtitle, fontsize=16, fontweight='bold', y=0.981)  
    # fig.suptitle(f"{df.metric[0]}  for {variable}  per  PERIOD ", fontsize=16, fontweight='bold', y=0.981)
    for j in range(i + 1, len(axe)):
        fig.delaxes(axe[j])
    file_out=f"{df.metric[0]}_{variable}_{zone}.png"
    plt.savefig(f'/home/mohamed/EHTPIII/MODELISATION/Report_25_11/plots/det/{df.metric[0]}/{file_out}')
    # plt.savefig(f'/home/mohamed/EHTPIII/MODELISATION/REPORT/Report_25_11/plots/det/{df.metric[0]}/{df.metric[0]}_{variable}.png',dpi=350)
        
    plt.tight_layout()
    plt.show()       

file_mena=[acc_df,corr_df,rsquared_df,rmse_df]
file_northafrica=[acc_df_NorthAfrica,corr_df_NorthAfrica,rsquared_df_NorthAfrica,rmse_df_NorthAfrica]
file_arabianpeninsula=[acc_df_ArabianPeninsula,corr_df_ArabianPeninsula,rsquared_df_ArabianPeninsula,rmse_df_ArabianPeninsula]
# for mask , df in zip([False]*3+[True]*3,file_list):
#     plot_determinist(df,"RR",mask)
    
# for mask , df in zip([False]*3+[True]*3,file_list):
#     plot_determinist(df,"T2M",mask)
    
for df in file_mena:
    for var in ["T2M","RR"]:
        plot_determinist(df,var,False,"mena")
        
for df in file_northafrica:
    for var in ["T2M","RR"]:
        plot_determinist(df,var,True,"NorthAfrica")
        
for df in file_arabianpeninsula:
    for var in ["T2M","RR"]:
        plot_determinist(df,var,True,"ArabianPeninsula")