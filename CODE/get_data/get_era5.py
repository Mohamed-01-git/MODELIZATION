#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:38:33 2024

@author: mohamed
"""

import cdsapi
import argparse
from osop.constants import SYSTEMS
import numpy as np





CDSAPI_URL = 'https://cds-beta.climate.copernicus.eu/api'
CDSAPI_KEY = '328ae3b8-36e2-4c27-9892-c7a55f386cdf'
c = cdsapi.Client(url=CDSAPI_URL, key=CDSAPI_KEY)
DATADIR ="/home/mohamed/EHTPIII/MODELISATION/DATA"   
    
obs_fname = '/home/mohamed/EHTPIII/MODELISATION/DATA/era5_monthly_stmonth{"NOVEMBER"}_{1993}-{2016}.grib'

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': [
            "2m_temperature",
            "total_precipitation",
        ],
        'year': [str(year) for year in range(1993, 2017)],  # List of years as strings
        'month': ['{:02d}'.format(month) for month in range(1, 13)],  # All months from January to December as zero-padded strings
        'time': '00:00',
        'grid': '1/1',  # Interpolating ERA5 data to 1x1 degree grid
        'area': [45, -30, -2.5, 60],  # North, West, South, East coordinates
        'format': 'grib',
    },
    obs_fname
)
