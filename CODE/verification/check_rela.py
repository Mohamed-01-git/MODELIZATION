#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 23:46:15 2024

@author: mohamed
"""

# CDS API

import cdsapi



# Libraries for working with multi-dimensional arrays

import xarray as xr

import pandas as pd

import numpy as np

import os

# Forecast verification metrics with xarray

import xskillscore as xs



# Date and calendar libraries

from dateutil.relativedelta import relativedelta

import calendar



# Libraries for plotting and geospatial data visualisation

from matplotlib import pyplot as plt

import cartopy.crs as ccrs

import cartopy.feature as cfeature



# Disable warnings for data download via API and matplotlib (do I need both???)

import warnings
import re
import os

warnings.filterwarnings('ignore')



file_path = "/home/mohamed/EHTPIII/MODELISATION/DATA/DATASET/IN/rea/era5_monthly_stmonth_RR_1993_2016.grib"
era5_1deg = xr.open_dataset(file_path, engine="cfgrib")
era5_1deg


era5_1deg = era5_1deg.rename({'latitude':'lat','longitude':'lon','time':'start_date'}).swap_dims({'start_date':'valid_time'})
valid_time = pd.to_datetime(era5_1deg.valid_time)
valid_time_normalized = valid_time.normalize()
era5_1deg["valid_time"]=valid_time_normalized
era_anom=era5_1deg-era5_1deg.mean("valid_time",skipna=True)



DATAIN ="/home/mohamed/EHTPIII/MODELISATION/DATA/DATASET/IN/grib"
    
DATAOUT="/home/mohamed/EHTPIII/MODELISATION/DATA/DATASET/OUT_2"


file="/ukmo_602_1993-2016_monthly_mean_11_234_45_-30_-2.5_60_djf.grib"

match = re.search(r"monthly_mean_(\d+)_", file)
start=int(match.group(1))

config = dict(

    list_vars = ['tp', ],

    hcstarty = 1993,

    hcendy = 2016,

    start_month = start,

)

import os


SCOREDIR = DATAOUT + '/SF/scores'

PLOTSDIR = DATAOUT + f'/SF/plots/stmonth{config["start_month"]:02d}'



for directory in [DATAIN, SCOREDIR, PLOTSDIR]:

    # Check if the directory exists

    if not os.path.exists(directory):

        # If it doesn't exist, create it

        os.makedirs(directory)

        print(f'Creating folder {directory}')
st_dim_name = 'time' if not config.get('isLagged',False) else 'indexing_time'

print('Reading HCST data from file')

# available_files = ["ukmo_602", "meteo_france_8", "ecmwf_51", "eccc_3", "eccc_2", "dwd_21", "cmcc_35"]


hcst_fname=DATAIN + f'/{file}'

hcst_bname=file.split(".grib")[0]
if "ecmwf" in hcst_bname:
    hcst = xr.open_dataset(hcst_fname,engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', st_dim_name)),drop_variables="t2m")
else:
    hcst = xr.open_dataset(hcst_fname,engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', st_dim_name)),drop_variables="p167")

time_interval_seconds = 30 * 86400 
hcst.tprate.values = hcst.tprate.values*time_interval_seconds*1000
hcst.attrs['units'] = 'mm'
hcst=hcst.rename({"tprate":"tp"})



hcst = hcst.chunk({'forecastMonth':1, 'latitude':'auto', 'longitude':'auto'})  #force dask.array using chunks on leadtime, latitude and longitude coordinate

hcst = hcst.rename({'latitude':'lat','longitude':'lon', st_dim_name:'start_date'})

print ('Re-arranging time metadata in xr.Dataset object')

# Add start_month to the xr.Dataset

start_month = pd.to_datetime(hcst.start_date.values[0]).month

hcst = hcst.assign_coords({'start_month':start_month})

# Add valid_time to the xr.Dataset

vt = xr.DataArray(dims=('start_date','forecastMonth'), coords={'forecastMonth':hcst.forecastMonth,'start_date':hcst.start_date})

vt.data = [[pd.to_datetime(std)+relativedelta(months=fcmonth-1) for fcmonth in vt.forecastMonth.values] for std in vt.start_date.values]

hcst = hcst.assign_coords(valid_time=vt)



# CALCULATE 3-month AGGREGATIONS

# NOTE rolling() assigns the label to the end of the N month period, so the first N-1 elements have NaN and can be dropped

print('Computing 3-month aggregation')

hcst_3m = hcst.rolling(forecastMonth=3,min_periods=1).mean(skipna=True)

hcst_3m = hcst_3m.where(hcst_3m.forecastMonth>=3,drop=True)





# CALCULATE ANOMALIES (and save to file)

print('Computing anomalies 1m')

hcmean = hcst.mean(['number','start_date'],skipna=True)

anom = hcst - hcmean

anom = anom.assign_attrs(reference_period='{hcstarty}-{hcendy}'.format(**config))



hcst_2=hcst.assign_attrs(reference_period='{hcstarty}-{hcendy}'.format(**config))

hcst_2_3m=hcst_2.rolling(forecastMonth=3,min_periods=1).mean(skipna=True)

hcst_2_3m = hcst_2_3m.where(hcst_2_3m.forecastMonth>=3,drop=True)



print('Computing anomalies 3m')

hcmean_3m = hcst_3m.mean(['number','start_date'],skipna=True)

anom_3m = hcst_3m - hcmean_3m

anom_3m = anom_3m.assign_attrs(reference_period='{hcstarty}-{hcendy}'.format(**config))



print('Saving anomalies 1m/3m to netCDF files')

# anom.to_netcdf(f'{DATAOUT}/{hcst_bname}.1m.RR.anom.nc')

# hcst_2.to_netcdf(f'{DATAOUT}/{hcst_bname}.1m.RR.hcst_2.nc')

# hcst_2_3m.to_netcdf(f'{DATAOUT}/{hcst_bname}.3m.RR.hcst_2.nc')

# anom_3m.to_netcdf(f'{DATAOUT}/{hcst_bname}.3m.RR.anom.nc')
# We define a function to calculate the boundaries of forecast categories defined by quantiles

def get_thresh(icat,quantiles,xrds,dims=['number','start_date']):



    if not all(elem in xrds.dims for elem in dims):           

        raise Exception('Some of the dimensions in {} is not present in the xr.Dataset {}'.format(dims,xrds)) 

    else:

        if icat == 0:

            xrds_lo = -np.inf

            xrds_hi = xrds.quantile(quantiles[icat],dim=dims)      

            

        elif icat == len(quantiles):

            xrds_lo = xrds.quantile(quantiles[icat-1],dim=dims)

            xrds_hi = np.inf

            

        else:

            xrds_lo = xrds.quantile(quantiles[icat-1],dim=dims)

            xrds_hi = xrds.quantile(quantiles[icat],dim=dims)

      

    return xrds_lo,xrds_hi

print('Computing probabilities (tercile categories)')

quantiles = [1/3., 2/3.]

numcategories = len(quantiles)+1



for aggr,h in [("1m",hcst), ("3m",hcst_3m)]:

    print(f'Computing tercile probabilities {aggr}')



    l_probs_hcst=list()

    for icat in range(numcategories):

        print(f'category={icat}')

        h_lo,h_hi = get_thresh(icat, quantiles, h)

        probh = np.logical_and(h>h_lo, h<=h_hi).sum('number',skipna=True)/float(h.dims['number'])

        # Instead of using the coordinate 'quantile' coming from the hindcast xr.Dataset

        # we will create a new coordinate called 'category'

        if 'quantile' in probh:

            probh = probh.drop('quantile')

        l_probs_hcst.append(probh.assign_coords({'category':icat}))



    print(f'Concatenating {aggr} tercile probs categories')

    probs = xr.concat(l_probs_hcst,dim='category')

    print(f'Saving {aggr} tercile probs netCDF files')

    # probs.to_netcdf(f'{DATAOUT}/{hcst_bname}.{aggr}.RR.tercile_probs.nc')



from os.path import join



# Loop over aggregations

for aggr in ['1m', '3m']:

    if aggr == '1m':

        o = era5_1deg

    elif aggr == '3m':

        o = era5_1deg

    else:

        raise BaseException(f'Unknown aggregation {aggr}')

    print(f'Computing deterministic scores for {aggr}-aggregation')

    # Read hindcast probabilities file

    probs_hcst = xr.open_dataset(f'{DATAOUT}/{hcst_bname}.{aggr}.RR.tercile_probs.nc')

    if "tp" not in probs_hcst.variables:
        probs_hcst=probs_hcst.rename({"p228":"tp"})

    

    l_roc = list()

    l_rps = list()

    l_rocss = list()

    l_bs = list()

    l_rela=list()


    for this_fcmonth in probs_hcst.forecastMonth.values:

        print(f'forecastMonth={this_fcmonth}')

        thishcst = probs_hcst.sel(forecastMonth=this_fcmonth).swap_dims({'start_date': 'valid_time'})

        # CALCULATE probabilities from observations

        print('We need to calculate probabilities (tercile categories) from observations')

        l_probs_obs = list()

        # Align both forecast and observation data on 'valid_time'

        thiso = o.where(o.valid_time == thishcst.valid_time, drop=True)

        thishcst_aligned, thiso_aligned = xr.align(thishcst, thiso, join='inner')

        for icat in range(3):

            # Compute category thresholds and probabilities for observations

            o_lo, o_hi = get_thresh(icat, quantiles, thiso_aligned, dims=['valid_time'])

            probo = 1. * np.logical_and(thiso_aligned > o_lo, thiso_aligned <= o_hi)

            if 'quantile' in probo:

                probo = probo.drop('quantile')

            l_probs_obs.append(probo.assign_coords({'category': icat}))

        thisobs = xr.concat(l_probs_obs, dim='category')

        # Now we can calculate the probabilistic (tercile categories) scores

        print('Now we can calculate the probabilistic (tercile categories) scores')

        thisroc = xr.Dataset()

        thisrps = xr.Dataset()

        thisrocss = xr.Dataset()

        thisbs = xr.Dataset()

        this_rela=xr.Dataset()



        for var in thishcst_aligned.data_vars:

            thisobs_binary = thisobs[var].astype(bool)
            forecast_probs = thishcst_aligned[var]
            this_rela[var] = xs.reliability(observations=thisobs_binary, forecasts=forecast_probs, probability_bin_edges =np.linspace(0,1,10),dim="valid_time")


        
        l_rela.append(this_rela)



    print('concat rela')

    rela=xr.concat(l_rela,dim='forecastMonth')



    print('writing to netcdf rela')

    # rela.to_netcdf(f'{SCOREDIR}/{hcst_bname}.{aggr}.RR.rela.nc')