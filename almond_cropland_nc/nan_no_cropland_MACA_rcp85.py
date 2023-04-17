import os
os.environ['PROJ_LIB'] = r'/home/shqwu/miniconda3/pkgs/proj4-5.2.0-he1b5a44_1006/share/proj'
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray
import pandas as pd
import numpy as np
import salem
import cartopy.crs as ccrs
import regionmask
import cartopy.feature as cfeature
from salem.utils import get_demo_file
from numpy import savetxt
import netCDF4 as nc

lat_with_cropland_sum = np.zeros((0))
lon_with_cropland_sum = np.zeros((0))
for year in range(2007,2022):
    cropland_nc = nc.Dataset('/home/shqwu/MACA/almond_cropland_nc/almond_cropland_'+str(year)+'.nc')
    cropland = cropland_nc.variables['almond'][:]
    cropland_lat = cropland_nc.variables['lat'][:]
    cropland_lon = cropland_nc.variables['lon'][:]
    lat_with_cropland = np.where(cropland!=0)[0]
    lon_with_cropland = np.where(cropland!=0)[1]
    lat_with_cropland_sum = np.concatenate((lat_with_cropland_sum, lat_with_cropland))
    lon_with_cropland_sum = np.concatenate((lon_with_cropland_sum, lon_with_cropland))

##obtain gridmet lat lon with almond
def find_nearest_cell(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

latarray = np.linspace(49.4,25.06666667,585) ##gridmet lat
lonarray = np.linspace(-124.76666663,-67.0583333,1386) ##gridmet lon

gridmet_almond_lat = np.zeros((lat_with_cropland_sum.shape[0]))
gridmet_almond_lon = np.zeros((lon_with_cropland_sum.shape[0]))

for i in range(0,lat_with_cropland_sum.shape[0]):
    gridmet_almond_lat[i] = find_nearest_cell(latarray, cropland_lat[lat_with_cropland_sum[i].astype(int)])
    gridmet_almond_lon[i] = find_nearest_cell(lonarray, cropland_lon[lon_with_cropland_sum[i].astype(int)])
gridmet_almond_lat_lon = np.row_stack((gridmet_almond_lat, gridmet_almond_lon)).astype(int)

## nan MACA nc
var_list = ['pr', 'tasmin', 'tasmax','huss', 'rhsmax', 'rhsmin', 'rsds', 'vpd', 'uas', 'vas']
var_name_list = ['precipitation', 'air_temperature', 'air_temperature', 'specific_humidity', 'relative_humidity', 'relative_humidity', 'surface_downwelling_shortwave_flux_in_air', 'vpd', 'eastward_wind', 'northward_wind']
model_list = ['bcc-csm1-1','bcc-csm1-1-m', 'BNU-ESM', 'CanESM2', 'CSIRO-Mk3-6-0', 'GFDL-ESM2G', 'GFDL-ESM2M', 'inmcm4', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR','CNRM-CM5', 'HadGEM2-CC365','HadGEM2-ES365', 'IPSL-CM5B-LR', 'MIROC5', 'MIROC-ESM', 'MIROC-ESM-CHEM']
period_list = ['2006_2010', '2011_2015', '2016_2020', '2021_2025','2026_2030', '2031_2035', '2036_2040', '2041_2045', '2046_2050', '2051_2055', '2056_2060', '2061_2065', '2066_2070', '2071_2075', '2076_2080', '2081_2085', '2086_2090', '2091_2095', '2096_2099']

for model in model_list:
    for k in range(0,1):
        for period in period_list:
            print(str(model))
            print(str(period))
            print(str(k))
            nc_data = nc.Dataset('/group/moniergrp/MACA/MACA_mask_no_almond/macav2metdata_'+str(var_list[k])+'_'+str(model)+'_r1i1p1_rcp85_'+str(period)+'_CONUS_daily.nc', 'r+')
            day_num = nc_data.variables[str(var_name_list[k])][:].shape[0]
            matrix = np.zeros((day_num, 585,1386))
            matrix[:] = np.nan
            for i in range(0,gridmet_almond_lat_lon.shape[1]):
                matrix[:,gridmet_almond_lat_lon[0,i],gridmet_almond_lat_lon[1,i]] = 1
            nc_data.variables[str(var_name_list[k])][:] = nc_data.variables[str(var_name_list[k])][:]*matrix
            nc_data.close()

