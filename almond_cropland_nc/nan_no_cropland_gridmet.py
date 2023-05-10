##code to filter out gridMET gridcells without almond cropland


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

home_path=
input_path_CDL = home_path+'/input_data/almond_cropland_nc/'
input_path_gridmet = home_path+'/input_data/Gridmet/'

lat_with_cropland_sum = np.zeros((0))
lon_with_cropland_sum = np.zeros((0))
for year in range(2007,2022):
    cropland_nc = nc.Dataset(input_path_CDL+'almond_cropland_'+str(year)+'.nc')
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

##create matrix with only copland lat-lon =1
matrix_365 = np.zeros((365, 585, 1386))
matrix_366 = np.zeros((366, 585, 1386))
matrix_365[:] = np.nan
matrix_366[:] = np.nan

for i in range(0,gridmet_almond_lat_lon.shape[1]):
    matrix_365[:,gridmet_almond_lat_lon[0,i],gridmet_almond_lat_lon[1,i]] = 1
    matrix_366[:,gridmet_almond_lat_lon[0,i],gridmet_almond_lat_lon[1,i]] = 1


## nan gridmet nc 
var_list = ['pet', 'sph', 'vs', 'pr', 'tmmx', 'tmmn']
var_name_list = ['potential_evapotranspiration', 'specific_humidity', 'wind_speed','precipitation_amount', 'air_temperature', 'air_temperature']

for k in range(0,6):
    for year in range(1979,2021):
        nc_data = nc.Dataset(input_path_gridmet+str(var_list[k])+'_'+str(year)+'.nc', 'r+')
        if nc_data.variables[str(var_name_list[k])][:].shape[0] == 366:
           nc_data.variables[str(var_name_list[k])][:] = nc_data.variables[str(var_name_list[k])][:]*matrix_366
        if nc_data.variables[str(var_name_list[k])][:].shape[0] == 365:
           nc_data.variables[str(var_name_list[k])][:] = nc_data.variables[str(var_name_list[k])][:]*matrix_365
        nc_data.close()


