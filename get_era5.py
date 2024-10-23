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
    
DATADIR = "/home/mohamed/EHTPIII/MODELISATION/DATA"  # Update with your actual data directory
config = {
    'list_vars': ['2m_temperature', 'total_precipitation'],  # Example variables, adjust as needed
    'obsstarty': 1993,  # Starting year
    'obsendy': 2016,  # Ending year
    'start_month': 1  # Start month (January)
}

# Create the output filename
obs_fname = '{fpath}/era5_monthly_stmonth{start_month:02d}_{hcstarty}-{hcendy}.grib'.format(
    fpath=DATADIR,
    start_month=config['start_month'],
    hcstarty=config['obsstarty'],
    hcendy=config['obsendy']
)

print(obs_fname)

# Retrieve data from the CDS
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': config['list_vars'],
        'year': ['{}'.format(yy) for yy in range(config['obsstarty'], config['obsendy'] + 2)],  # Including one extra year
        'month': ['{:02d}'.format((config['start_month'] + leadm - 1) % 12 + 1) for leadm in range(6)],  # Adjust month calculation
        'time': '00:00',
        'grid': '1/1',  # Interpolation grid
        'area': '45/-30/-2.5/60',  # Area defined by N,W,S,E
        'data_format': 'grib',
    },
    obs_fname
)