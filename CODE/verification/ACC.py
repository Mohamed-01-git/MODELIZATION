#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:19:13 2024

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
era5_1deg = era5_1deg.rename({'latitude':'lat','longitude':'lon','time':'start_date'}).swap_dims({'start_date':'valid_time'})
valid_time = pd.to_datetime(era5_1deg.valid_time)
valid_time_normalized = valid_time.normalize()
era5_1deg["valid_time"]=valid_time_normalized


config = dict(

    list_vars = ['tp', ],

    hcstarty = 1993,

    hcendy = 2016,

    start_month = 2,

)





era_anom=era5_1deg-era5_1deg.mean("valid_time")
# era_anom_3m=era_anom.rolling(time=3).mean()


fcmonths = [mm+1 if mm>=0 else mm+13 for mm in [t.month - config['start_month'] for t in pd.to_datetime(era5_1deg.valid_time.values)] ]

era5_1deg = era5_1deg.assign_coords(forecastMonth=('valid_time',fcmonths))
era_anom=era_anom.assign_coords(forecastMonth=('valid_time',fcmonths))



era5_1deg = era5_1deg.where(era5_1deg.valid_time>=np.datetime64('{hcstarty}-{start_month:02d}-01'.format(**config)),drop=True)
era_anom = era_anom.where(era_anom.valid_time>=np.datetime64('{hcstarty}-{start_month:02d}-01'.format(**config)),drop=True)
era_anom_3m= era_anom.rolling(valid_time=3,min_periods=1).mean(skipna=True)
era_anom_3m=era_anom_3m.where(era_anom_3m.forecastMonth>=3)

era_anom = era_anom.drop('forecastMonth')

era_anom_3m = era_anom_3m.drop('forecastMonth')

o = era_anom

h=xr.open_dataset("/home/mohamed/EHTPIII/MODELISATION/DATA/DATASET/OUT/ukmo_602_1993-2016_monthly_mean_2_234_45_-30_-2.5_60_mam.1m.RR.anom.nc")

is_fullensemble = 'number' in h.dims
l_corr=[]
for this_fcmonth in h.forecastMonth.values:
    print(f'forecastMonth={this_fcmonth}')
    thishcst = h.sel(forecastMonth=this_fcmonth).swap_dims({'start_date': 'valid_time'})

    thisobs = o.where(o.valid_time == thishcst.valid_time, drop=True)

    # Align the forecast and observation data along all common dimensions
    thishcst_em, thisobs_aligned = xr.align(thishcst, thisobs, join='inner')

    # If it's a full ensemble, take the mean over the 'number' dimension
    thishcst_em = thishcst_em if not is_fullensemble else thishcst_em.mean('number',skipna=True)

    l_corr.append(xs.spearman_r(thishcst_em, thisobs_aligned, dim='valid_time'))
    
    
corr = xr.concat(l_corr, dim='forecastMonth')