##code to create a reference MACA nc file masked by CDL
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

home_path='~/Run_project'

lat_with_cropland_sum = np.zeros((0))
lon_with_cropland_sum = np.zeros((0))
for year in range(2007,2021):
    cropland_nc = nc.Dataset(home_path+'/input_data/almond_cropland_nc/almond_cropland_'+str(year)+'.nc')
    cropland = cropland_nc.variables['almond'][:]
    cropland_lat = cropland_nc.variables['lat'][:]
    cropland_lon = cropland_nc.variables['lon'][:]
    lat_with_cropland = np.where(cropland!=0)[0]
    lon_with_cropland = np.where(cropland!=0)[1]
    lat_with_cropland_sum = np.concatenate((lat_with_cropland_sum, lat_with_cropland))
    lon_with_cropland_sum = np.concatenate((lon_with_cropland_sum, lon_with_cropland))

##obtain maca lat lon with almond
def find_nearest_cell(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


##Gridmet
latarray = np.linspace(49.4,25.06666667,585) ##gridmet lat
lonarray = np.linspace(-124.76666663,-67.0583333,1386) ##gridmet lon

gridmet_almond_lat = np.zeros((lat_with_cropland_sum.shape[0]))
gridmet_almond_lon = np.zeros((lon_with_cropland_sum.shape[0]))

for i in range(0,lat_with_cropland_sum.shape[0]):
    gridmet_almond_lat[i] = find_nearest_cell(latarray, cropland_lat[lat_with_cropland_sum[i].astype(int)])
    gridmet_almond_lon[i] = find_nearest_cell(lonarray, cropland_lon[lon_with_cropland_sum[i].astype(int)])
gridmet_almond_lat_lon = np.row_stack((gridmet_almond_lat, gridmet_almond_lon)).astype(int)

nc_data = nc.Dataset(home_path+'/input_data/reference_cropland/+'tmmn_1979.nc','r+')
day_num = nc_data.variables['air_temperature'].shape[0]
matrix = np.zeros((day_num, 585,1386))
matrix[:] = np.nan
for i in range(0,gridmet_almond_lat_lon.shape[1]):
    matrix[:,gridmet_almond_lat_lon[0,i],gridmet_almond_lat_lon[1,i]] = 1
nc_data.variables['air_temperature'][:] = nc_data.variables['air_temperature'][:]*matrix
nc_data.close()


##MACA
latarray = np.linspace(25.06666667,49.4,585) ##maca lat
lonarray = np.linspace(-124.76666663,-67.0583333,1386) ##maca lon

maca_almond_lat = np.zeros((lat_with_cropland_sum.shape[0]))
maca_almond_lon = np.zeros((lon_with_cropland_sum.shape[0]))

for i in range(0,lat_with_cropland_sum.shape[0]):
    maca_almond_lat[i] = find_nearest_cell(latarray, cropland_lat[lat_with_cropland_sum[i].astype(int)])
    maca_almond_lon[i] = find_nearest_cell(lonarray, cropland_lon[lon_with_cropland_sum[i].astype(int)])
maca_almond_lat_lon = np.row_stack((maca_almond_lat, maca_almond_lon)).astype(int)

nc_data = nc.Dataset(home_path+'/input_data/reference_cropland/macav2metdata_tasmin_bcc-csm1-1_r1i1p1_historical_1950_1954_CONUS_daily.nc', 'r+')
day_num = nc_data.variables[str(var_name_list[k])].shape[0]
matrix = np.zeros((day_num, 585,1386))
matrix[:] = np.nan
for i in range(0,maca_almond_lat_lon.shape[1]):
    matrix[:,maca_almond_lat_lon[0,i],maca_almond_lat_lon[1,i]] = 1
nc_data.variables[str(var_name_list[k])][:] = nc_data.variables[str(var_name_list[k])][:]*matrix
nc_data.close()



##LOCA
latarray = np.linspace(29.578125,45.015625,495) ##LOCA lat
lonarray = np.linspace(-128.42188,-110.984375,559) ##LOCA lon

gridmet_almond_lat = np.zeros((lat_with_cropland_sum.shape[0]))
gridmet_almond_lon = np.zeros((lon_with_cropland_sum.shape[0]))

for i in range(0,lat_with_cropland_sum.shape[0]):
    gridmet_almond_lat[i] = find_nearest_cell(latarray, cropland_lat[lat_with_cropland_sum[i].astype(int)])
    gridmet_almond_lon[i] = find_nearest_cell(lonarray, cropland_lon[lon_with_cropland_sum[i].astype(int)])
gridmet_almond_lat_lon = np.row_stack((gridmet_almond_lat, gridmet_almond_lon)).astype(int)

## Because the each nc file of the original LOCA data is so large, instead of masking the entire nc file, I just extract one day 
##from a temperature LOCA file as the reference LOCA nc
nc_data = nc.Dataset('home_path+'/input_data/reference_cropland/LOCA_reference_cropland.nc','r+')
matrix = np.zeros((495,559))
matrix[:] = np.nan
for i in range(0,gridmet_almond_lat_lon.shape[1]):
    matrix[gridmet_almond_lat_lon[0,i],gridmet_almond_lat_lon[1,i]] = 1
nc_data.variables['tasmax'][:] = nc_data.variables['tasmax'][:]*matrix
nc_data.close()



