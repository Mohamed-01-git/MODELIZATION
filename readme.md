# Seasonal Forecasting Project

This project involves downloading necessary data, calculating scores for various climate variables, and plotting the results for seasonal forecasting. The project is organized into several key folders:

## Project Structure

- **Report_25_11**: This folder contains the Report.
- **ADVANCEMENT_DEC**: This folder contains the presentation materials for the last checkpoint.
- **CODE**: Contains all the scripts and codes used for downloading data, calculating scores, and plotting results.

- **readme.md**: This readme file.

## Requirements

Make sure to have the following dependencies installed to run the scripts:

- Python (>=3.7)
- Libraries:
  - `xarray`
  - `matplotlib`
  - `numpy`
  - `pandas`
  - `cartopy`
  - `seaborn`
  - `pygrib` (for `.grib` file handling)
  - `osop`
  - `cfgrib`
  - `cdsapi`

## Getting Started

1. **Download Data**: 
   - Download and preprocess the `.grib` files located in the `DATASETS` folder. These files contain climate data for the seasonal forecast models.
   - Use the Python code in the `CODE` folder to process the `.grib` files (get_data).

2. **Calculate Scores**:
   - The `SFscores` folder contains pre-calculated scores for the seasonal forecast models (`s-f-rr.ipynb`), including TPRATE (Total Precipitation Rate), TP (Total Precipitation), and T2M (Temperature at 2 meters).
   - These scores are used to assess the accuracy of the model forecasts.

3. **Plotting Results**:
   - The `CODE` folder also includes scripts (`sf-plot-scores.ipynb`) to generate various plots, including seasonal forecast maps, and the results of the forecast models.
   - Maps can be generated using the shapefiles in the `shapefile_morocco_world` folder.

## Running the Code

### Downloading Data
To download the required data, run the following script:

```bash
`get_arabcof_{period}.sh`
