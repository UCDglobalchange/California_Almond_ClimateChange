## Code to calculate ACIs for each MACA model

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
import math

data_ID='11_19'
model_name='MRI-CGCM3'
save_path = '/home/shqwu/Almond_code_git/saved_data/'+str(data_ID)+'/MACA_ACI/'+str(model_name)+'/'
county_list = ['Butte', 'Colusa', 'Fresno', 'Glenn', 'Kern', 'Kings', 'Madera', 'Merced', 'San Joaquin', 'Solano', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Yolo', 'Yuba']                      
shapefile = salem.read_shapefile('/home/pgzikala/Shapefiles/CA_Counties/Counties.shp')
for county in county_list:
     locals()[str(county)+'_shp'] = shapefile.loc[shapefile['NAME'].isin([str(county)])]

period_list = ['1950_1954', '1955_1959', '1960_1964', '1965_1969','1970_1974', '1975_1979', '1980_1984', '1985_1989', '1990_1994', '1995_1999', '2000_2004', '2005_2005']

var_list = ['pr', 'tasmin', 'tasmax','huss', 'rhsmax', 'rhsmin', 'rsds', 'vpd', 'uas', 'vas']
cropland_reference = salem.open_xr_dataset('/group/moniergrp/MACA/MACA_mask_no_almond/macav2metdata_tasmin_bcc-csm1-1_r1i1p1_historical_1950_1954_CONUS_daily.nc')
for county in county_list:
    locals()[str(county)+'_reference'] = cropland_reference.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0).salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False).air_temperature
for var in var_list:
    print(var)
    for period in period_list:
        nc_change_lon = nc.Dataset('/group/moniergrp/MACA/macav2metdata_'+str(var)+'_'+str(model_name)+'_r1i1p1_historical_'+str(period)+'_CONUS_daily.nc', 'r+')
        if nc_change_lon.variables['lon'][0]+0<-800:
            nc_change_lon.variables['lon'][:] = nc_change_lon.variables['lon'][:]+720
        elif nc_change_lon.variables['lon'][0]+0<-400:
            nc_change_lon.variables['lon'][:] = nc_change_lon.variables['lon'][:]+360
        elif nc_change_lon.variables['lon'][0]+0<-200:
            pass
        elif nc_change_lon.variables['lon'][0]+0>0:
            nc_change_lon.variables['lon'][:] = nc_change_lon.variables['lon'][:]-360
        nc_change_lon.close()
        locals()[str(var)+str(period)] = salem.open_xr_dataset('/group/moniergrp/MACA/macav2metdata_'+str(var)+'_'+str(model_name)+'_r1i1p1_historical_'+str(period)+'_CONUS_daily.nc')
        for county in county_list:
            if np.int(np.int(period)/10000) != 2005:
                year_1 = np.int(np.int(period)/10000)
                for year in range(year_1, year_1+5):
                    if var == 'uas' or var == 'vas':
                        locals()[str(county)+str(var)+str(year)+'_roi'] = locals()[str(var)+str(period)].salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
                    else:
                        locals()[str(county)+str(var)+str(year)+'_roi'] = locals()[str(var)+str(period)].salem.subset(shape = locals()[str(county)+'_shp'], margin = 0).salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
            else:
                year = 2005
                if var == 'uas' or var == 'vas':
                   locals()[str(county)+str(var)+str(year)+'_roi'] = locals()[str(var)+str(period)].salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
                else:
                   locals()[str(county)+str(var)+str(year)+'_roi'] = locals()[str(var)+str(period)].salem.subset(shape = locals()[str(county)+'_shp'], margin = 0).salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)


##Bloom windspeed
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    print(year)
    for county in county_list:
        print(county)
        roivas = locals()[str(county)+'vas'+str(year)+'_roi']
        roiuas = locals()[str(county)+'uas'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roivas.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roivas.time.values).year==year)[0][-1]
        roivas = roivas.northward_wind[day_start:(day_end+1)]
        roiuas = roiuas.eastward_wind[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roivas.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        roivas = roivas.values[fall_start:fall_end+1]
        roiuas = roiuas[fall_start:fall_end+1].values
        sumdays = np.zeros((roivas.shape[0], roivas.shape[1], roivas.shape[2]))
        wpd = (roivas**2+roiuas**2)**0.5
        for day in range(0, roivas.shape[0]):
            for lat in range(0,roivas.shape[1]):
                for lon in range(0,roivas.shape[2]):
                    if wpd[day,lat,lon] > 6.6944:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1950,0] = year
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_windspeed.csv', locals()[str(county)+'_sum'], delimiter = ',')

## Bloom >12.8
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        sumdays = np.zeros((datatmax.shape[0], datatmax.shape[1], datatmax.shape[2]))
        for day in range(0, datatmax.shape[0]):
            for lat in range(0,datatmax.shape[1]):
                for lon in range(0,datatmax.shape[2]):
                    if (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 > 12.8:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1950,0] = year
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_T12.8.csv', locals()[str(county)+'_sum'], delimiter = ',')


##Bloom >15.6
print('JanFeb >15.6')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        sumdays = np.zeros((datatmax.shape[0], datatmax.shape[1], datatmax.shape[2]))
        for day in range(0, datatmax.shape[0]):
            for lat in range(0,datatmax.shape[1]):
                for lon in range(0,datatmax.shape[2]):
                    if (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 > 15.6:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan        
        locals()[str(county)+'_sum'][year-1950,0] = year           
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_T15.6.csv', locals()[str(county)+'_sum'], delimiter = ',')


##bloom 18.3-26.7
print('JanFeb >=18.3 <= 26.7')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        sumdays = np.zeros((datatmax.shape[0], datatmax.shape[1], datatmax.shape[2]))
        for day in range(0, datatmax.shape[0]):
            for lat in range(0,datatmax.shape[1]):
                for lon in range(0,datatmax.shape[2]):
                    if (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 >= 18.3 and (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 <= 26.7:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan        
        locals()[str(county)+'_sum'][year-1950,0] = year           
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_T18.3-26.7.csv', locals()[str(county)+'_sum'], delimiter = ',')


##Bloom 10-21.1
print('JanFeb >=10 <= 21.1')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        sumdays = np.zeros((datatmax.shape[0], datatmax.shape[1], datatmax.shape[2]))
        for day in range(0, datatmax.shape[0]):
            for lat in range(0,datatmax.shape[1]):
                for lon in range(0,datatmax.shape[2]):
                    if (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 >= 10 and (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 <= 21.1:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan        
        locals()[str(county)+'_sum'][year-1950,0] = year           
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_T10_21.1.csv', locals()[str(county)+'_sum'], delimiter = ',')


##Bloom 21.1-30.6
print('JanFeb >=21.1 <= 30.6')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        sumdays = np.zeros((datatmax.shape[0], datatmax.shape[1], datatmax.shape[2]))
        for day in range(0, datatmax.shape[0]):
            for lat in range(0,datatmax.shape[1]):
                for lon in range(0,datatmax.shape[2]):
                    if (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 >= 21.1 and (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 <= 30.6:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan        
        locals()[str(county)+'_sum'][year-1950,0] = year           
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_T21.1-30.6.csv', locals()[str(county)+'_sum'], delimiter = ',')


##Bloom >4.4
print('JanFeb >4.4')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        sumdays = np.zeros((datatmax.shape[0], datatmax.shape[1], datatmax.shape[2]))
        for day in range(0, datatmax.shape[0]):
            for lat in range(0,datatmax.shape[1]):
                for lon in range(0,datatmax.shape[2]):
                    if (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 > 4.4:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan        
        locals()[str(county)+'_sum'][year-1950,0] = year           
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_T4.4.csv', locals()[str(county)+'_sum'], delimiter = ',')



    

print('Mar Tmin')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    for county in county_list:
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][-1]
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmin.values.shape[0] == 366:
            fall_start = 60
            fall_end = 90
        else:
            fall_start = 59
            fall_end = 89
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-1950,0] = year           
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(Tmindata)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_MarTmin.csv', locals()[str(county)+'_sum'], delimiter = ',')

##Dormancy chill
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1951, 2006):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata =  locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        roitmax = locals()[str(county)+'tasmax'+str(year-1)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = roitmax.values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        pre_Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year-1)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        pre_Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        WC = np.zeros((Tmindata.shape[0], Tmindata.shape[1],Tmindata.shape[2]))
        pre_WC = np.zeros((pre_Tmindata.shape[0], Tmindata.shape[1],Tmindata.shape[2]))
        for day in range(0, Tmindata.shape[0]):
            for lat in range(0, Tmindata.shape[1]):
                for lon in range(0, Tmindata.shape[2]):
                    if Tmindata[day,lat,lon] >7.22:
                       WC[day,lat,lon] = 0
                    elif Tmindata[day,lat,lon] >0:
                       WC[day,lat,lon] = ((6/((Tmindata[day,lat,lon]+Tmaxdata[day,lat,lon])/2-Tmindata[day,lat,lon]))*(7.22-Tmindata[day,lat,lon]))*2
                    else:
                       WC[day,lat,lon] = ((6/((Tmindata[day,lat,lon]+Tmaxdata[day,lat,lon])/2-Tmindata[day,lat,lon]))*(7.22-Tmindata[day,lat,lon]))*2-((6/((Tmindata[day,lat,lon]+Tmaxdata[day,lat,lon])/2-Tmindata[day,lat,lon]))*(0-Tmindata[day,lat,lon]))*2
        for day in range(0, pre_Tmindata.shape[0]):
            for lat in range(0, Tmindata.shape[1]):
                for lon in range(0, Tmindata.shape[2]):
                    if pre_Tmindata[day,lat,lon] >7.22:
                       pre_WC[day,lat,lon] = 0
                    elif pre_Tmindata[day,lat,lon] >0:
                       pre_WC[day,lat,lon] = ((6/((pre_Tmindata[day,lat,lon]+pre_Tmaxdata[day,lat,lon])/2-pre_Tmindata[day,lat,lon]))*(7.22-pre_Tmindata[day,lat,lon]))*2
                    else:
                       pre_WC[day,lat,lon] = ((6/((pre_Tmindata[day,lat,lon]+pre_Tmaxdata[day,lat,lon])/2-pre_Tmindata[day,lat,lon]))*(7.22-pre_Tmindata[day,lat,lon]))*2-((6/((pre_Tmindata[day,lat,lon]+pre_Tmaxdata[day,lat,lon])/2-pre_Tmindata[day,lat,lon]))*(0-pre_Tmindata[day,lat,lon]))*2

        WC[np.isinf(WC)] = np.nan
        pre_WC[np.isinf(pre_WC)] = np.nan
        WC[np.where(WC>2880)] = np.nan
        pre_WC[np.where(pre_WC>2880)] = np.nan
        WC = np.nansum(WC, axis = 0)
        pre_WC = np.nansum(pre_WC, axis = 0)
        WC = WC + pre_WC
        for lat in range(0,WC.shape[0]):
            for lon in range(0, WC.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    WC[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1950,0] = year
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(WC)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_DormancyChill.csv', locals()[str(county)+'_sum'], delimiter = ',')



##DormancyFreeze
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1951, 2006):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        roitmax = locals()[str(county)+'tasmax'+str(year-1)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = roitmax.values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        pre_Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year-1)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        pre_Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        WC = np.zeros((Tmindata.shape[0], Tmindata.shape[1],Tmindata.shape[2]))
        pre_WC = np.zeros((pre_Tmindata.shape[0], Tmindata.shape[1],Tmindata.shape[2]))
        for day in range(0, Tmindata.shape[0]):
            for lat in range(0, Tmindata.shape[1]):
                for lon in range(0, Tmindata.shape[2]):
                    if Tmindata[day,lat,lon] > 0:
                       WC[day,lat,lon] = 0
                    else:
                       WC[day,lat,lon] = ((6/((Tmindata[day,lat,lon]+Tmaxdata[day,lat,lon])/2-Tmindata[day,lat,lon]))*(0-Tmindata[day,lat,lon]))*2
        for day in range(0, pre_Tmindata.shape[0]):
            for lat in range(0, Tmindata.shape[1]):
                for lon in range(0, Tmindata.shape[2]):
                    if pre_Tmindata[day,lat,lon] > 0:
                       pre_WC[day,lat,lon] = 0
                    else:
                       pre_WC[day,lat,lon] = ((6/((pre_Tmindata[day,lat,lon]+pre_Tmaxdata[day,lat,lon])/2-pre_Tmindata[day,lat,lon]))*(0-pre_Tmindata[day,lat,lon]))*2
        WC[np.isinf(WC)] = np.nan
        pre_WC[np.isinf(pre_WC)] = np.nan
        WC[np.where(WC>2880)] = np.nan
        pre_WC[np.where(pre_WC>2880)] = np.nan
        WC = np.nansum(WC, axis = 0)
        pre_WC = np.nansum(pre_WC, axis = 0)
        WC = WC + pre_WC
        for lat in range(0,WC.shape[0]):
            for lon in range(0, WC.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    WC[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1950,0] = year
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(WC)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_DormancyFreeze.csv', locals()[str(county)+'_sum'], delimiter = ',')


##Dormancy ETo
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1951, 2006):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        lat = roitmax.lat.values
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata =  locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        huss = locals()[str(county)+'huss'+str(year)+'_roi'].specific_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #kg/kg
        rhsmax = locals()[str(county)+'rhsmax'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #%
        rhsmin = locals()[str(county)+'rhsmin'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values
        rsds =( locals()[str(county)+'rsds'+str(year)+'_roi'].surface_downwelling_shortwave_flux_in_air[day_start:(day_end+1)][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
        vpd = locals()[str(county)+'vpd'+str(year)+'_roi'].vpd[day_start:(day_end+1)][fall_start:fall_end+1].values #kPa
        uas = locals()[str(county)+'uas'+str(year)+'_roi'].eastward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values # m/s at 10m
        vas = locals()[str(county)+'vas'+str(year)+'_roi'].northward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values

        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        u_2m = 0.748*((uas**2+vas**2)**0.5)
        Tmean = (Tmaxdata+Tmindata)/2
        D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
        j = np.pi*lat/180
        g = 0.066
        e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
        e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
        ea = (e0_min*rhsmax/100+e0_max*rhsmin/100)/2
        Ra = np.zeros((Tmaxdata.shape[0],Tmaxdata.shape[1],Tmaxdata.shape[2]))
        for day in range(0, Tmaxdata.shape[0]):
            for lon in range(0,Tmaxdata.shape[2]):
                J = fall_start+day
                dr = 1+0.033*np.cos(2*np.pi*J/365)
                d = 0.409*np.sin(2*np.pi*J/365-1.39)
                ws = np.arccos(-np.tan(j)*np.tan(d))
                Ra[day,:,lon] = (24*60/np.pi)*0.082*dr*(ws*np.sin(j)*np.sin(d)+np.cos(j)*np.cos(d)*np.sin(ws))
        Rns = 0.77*rsds
        Rso = 0.75*Ra
        Rnl = 4.903e-9*(((Tmaxdata+273.16)**4+(Tmindata+273.16)**4)/2)*(0.34-0.14*(ea**0.5))*(1.35*(rsds/Rso)-0.35)
        Rn = Rns-Rnl
        pre_ETo = (0.408*D*Rn+g*900*u_2m*vpd/(Tmean+273.16))/(D+g*(1+0.34*u_2m))
        roitmax = locals()[str(county)+'tasmax'+str(year-1)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = roitmax.values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15

        huss = locals()[str(county)+'huss'+str(year-1)+'_roi'].specific_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #kg/kg
        rhsmax = locals()[str(county)+'rhsmax'+str(year-1)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #%
        rhsmin = locals()[str(county)+'rhsmin'+str(year-1)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values
        rsds = (locals()[str(county)+'rsds'+str(year-1)+'_roi'].surface_downwelling_shortwave_flux_in_air[day_start:(day_end+1)][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
        vpd = locals()[str(county)+'vpd'+str(year-1)+'_roi'].vpd[day_start:(day_end+1)][fall_start:fall_end+1].values #kPa
        uas = locals()[str(county)+'uas'+str(year-1)+'_roi'].eastward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values # m/s at 10m
        vas = locals()[str(county)+'vas'+str(year-1)+'_roi'].northward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values

        roitmin = locals()[str(county)+'tasmin'+str(year-1)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        u_2m = 0.748*((uas**2+vas**2)**0.5)
        Tmean = (Tmaxdata+Tmindata)/2
        D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
        j = np.pi*lat/180
        g = 0.066
        e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
        e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
        ea = (e0_min*rhsmax/100+e0_max*rhsmin/100)/2
        Ra = np.zeros((Tmaxdata.shape[0],Tmaxdata.shape[1],Tmaxdata.shape[2]))
        for day in range(0, Tmaxdata.shape[0]):
            for lon in range(0,Tmaxdata.shape[2]):
                J = fall_start+day
                dr = 1+0.033*np.cos(2*np.pi*J/365)
                d = 0.409*np.sin(2*np.pi*J/365-1.39)
                ws = np.arccos(-np.tan(j)*np.tan(d))
                Ra[day,:,lon] = (24*60/np.pi)*0.082*dr*(ws*np.sin(j)*np.sin(d)+np.cos(j)*np.cos(d)*np.sin(ws))

        Rns = 0.77*rsds
        Rso = 0.75*Ra
        Rnl = 4.903e-9*(((Tmaxdata+273.16)**4+(Tmindata+273.16)**4)/2)*(0.34-0.14*(ea**0.5))*(1.35*(rsds/Rso)-0.35)
        Rn = Rns-Rnl
        ETo = (0.408*D*Rn+g*900*u_2m*vpd/(Tmean+273.16))/(D+g*(1+0.34*u_2m))

        ETo_sum = np.row_stack((pre_ETo,ETo))
        ETo_day_num = ETo_sum.shape[0]
        ETo_sum = np.nanmean(ETo_sum,axis = 0)
        for lat in range(0,ETo_sum.shape[0]):
            for lon in range(0, ETo_sum.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    ETo_sum[lat,lon] = np.nan

        locals()[str(county)+'_sum'][year-1950,0] = year
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(ETo_sum)*ETo_day_num
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_DormancyETo.csv', locals()[str(county)+'_sum'], delimiter = ',')







print('Feb ppt')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2)) 
for year in range(1950, 2006):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roi.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roi.time.values).year==year)[0][-1]
        roi = roi.precipitation[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roi.values.shape[0] == 366:
            fall_start = 31
            fall_end = 59
        else:
            fall_start = 31
            fall_end = 58
        data = roi.values[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-1950,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_FebPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')
    
    
print('Feb Tmin')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    for county in county_list:
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][-1]
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmin.values.shape[0] == 366:
            fall_start = 31
            fall_end = 59
        else:
            fall_start = 31
            fall_end = 58
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-1950,0] = year           
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(Tmindata)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_FebTmin.csv', locals()[str(county)+'_sum'], delimiter = ',')
    
    


##jabfeb sph
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    for county in county_list:
        roitmin = locals()[str(county)+'huss'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][-1]
        roitmin = roitmin.specific_humidity[day_start:(day_end+1)]
        shapedata =  locals()[str(county)+'_reference'].values[1]
        if roitmin.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-1950,0] = year           
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(Tmindata)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_SpH.csv', locals()[str(county)+'_sum'], delimiter = ',')

##Jan Ppt
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2)) 
for year in range(1950, 2006):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roi.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roi.time.values).year==year)[0][-1]
        roi = roi.precipitation[day_start:(day_end+1)]
        shapedata = roi.values[1]
        if roi.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        data = roi.values[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-1950,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_JanPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')



##harvest ppt
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roi.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roi.time.values).year==year)[0][-1]
        roi = roi.precipitation[day_start:(day_end+1)]
        shapedata =  locals()[str(county)+'_reference'].values[1]
        if roi.values.shape[0] == 366:
            fall_start = 213
            fall_end = 304
        else:
            fall_start = 212
            fall_end = 303
        data = roi.values[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-1950,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_harvest_Ppt.csv', locals()[str(county)+'_sum'], delimiter = ',')


## harvest Tmean
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata =  locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 213
            fall_end = 304
        else:
            fall_start = 212
            fall_end = 303
        datatmax = roitmax.values[fall_start:fall_end+1]
        Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmaxdata[lat,lon] = np.nan
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-1950,0] = year           
        locals()[str(county)+'_sum'][year-1950,1] = (np.nanmean(Tmindata)+np.nanmean(Tmaxdata))/2
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_harvest_Tmean.csv', locals()[str(county)+'_sum'], delimiter = ',')



#Bloom Ppt
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2)) 
for year in range(1950, 2006):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roi.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roi.time.values).year==year)[0][-1]
        roi = roi.precipitation[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roi.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        data = roi.values[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-1950,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_BloomPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')

## Bloom Tmean
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]
        Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmaxdata[lat,lon] = np.nan
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-1950,0] = year           
        locals()[str(county)+'_sum'][year-1950,1] = (np.nanmean(Tmindata)+np.nanmean(Tmaxdata))/2
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_BloomTmean.csv', locals()[str(county)+'_sum'], delimiter = ',')
    
#Growing Ppt
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2)) 
for year in range(1950, 2006):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roi.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roi.time.values).year==year)[0][-1]
        roi = roi.precipitation[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roi.values.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        data = roi.values[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-1950,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_GrowingPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')
    
## Growing Tmean
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        datatmax = roitmax.values[fall_start:fall_end+1]
        Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmaxdata[lat,lon] = np.nan
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-1950,0] = year           
        locals()[str(county)+'_sum'][year-1950,1] = (np.nanmean(Tmindata)+np.nanmean(Tmaxdata))/2
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_GrowingTmean.csv', locals()[str(county)+'_sum'], delimiter = ',')
    

##Dormancy Ppt
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1951, 2006):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year-1)+'_roi']
        day_start = np.where(pd.to_datetime(roi.time.values).year==year-1)[0][0]
        day_end = np.where(pd.to_datetime(roi.time.values).year==year-1)[0][-1]
        roi = roi.precipitation[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roi.values.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        data = roi.values[fall_start:fall_end+1]
        pre_data = np.nansum(data, axis = 0)

        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roi.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roi.time.values).year==year)[0][-1]
        roi = roi.precipitation[day_start:(day_end+1)]
        if roi.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        data = roi.values[fall_start:fall_end+1]
        data = np.nansum(data, axis = 0)
        data = data+pre_data
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1950,0] = year
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_DormancyPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')


## Dormancy Tmean
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1951, 2006):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata =  locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        datatmax = roitmax.values[fall_start:fall_end+1]
        Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmaxdata[lat,lon] = np.nan
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan

        roitmax = locals()[str(county)+'tasmax'+str(year-1)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = roitmax.values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        datatmax = roitmax.values[fall_start:fall_end+1]
        pre_Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    pre_Tmaxdata[lat,lon] = np.nan
        roitmin = locals()[str(county)+'tasmin'+str(year-1)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]
        pre_Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    pre_Tmindata[lat,lon] = np.nan
        Tmindata = np.row_stack((Tmindata, pre_Tmindata))
        Tmaxdata = np.row_stack((Tmaxdata, pre_Tmaxdata))
        locals()[str(county)+'_sum'][year-1950,0] = year
        locals()[str(county)+'_sum'][year-1950,1] = (np.nanmean(Tmindata)+np.nanmean(Tmaxdata))/2
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_DormancyTmean.csv', locals()[str(county)+'_sum'], delimiter = ',')




## Growing KDD30
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        DD = np.zeros((datatmin.shape[0],datatmin.shape[1], datatmin.shape[2]))
        for day in range(0,datatmin.shape[0]):
            for lat in range(0,datatmin.shape[1]):
                for lon in range(0, datatmin.shape[2]):
                    if np.isnan(datatmax[day, lat,lon]):
                        pass
                    else:
                        b = 30
                        if datatmax[day, lat,lon]<=b:
                           DD[day, lat,lon] = 0
                        elif datatmin[day, lat,lon] < b and datatmax[day, lat,lon]> b:
                           DD[day, lat, lon] = np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-b)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi
                        elif datatmin[day, lat,lon] >=b :
                           DD[day, lat, lon] = (datatmin[day, lat,lon]+datatmax[day, lat,lon])/2-b
        DD = np.nansum(DD, axis = 0)
        for lat in range(0,DD.shape[0]):
            for lon in range(0, DD.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    DD[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1950,0] = year
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(DD)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_GrowingKDD30.csv', locals()[str(county)+'_sum'], delimiter = ',')


## Bloom KDD30
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        DD = np.zeros((datatmin.shape[0],datatmin.shape[1], datatmin.shape[2]))
        for day in range(0,datatmin.shape[0]):
            for lat in range(0,datatmin.shape[1]):
                for lon in range(0, datatmin.shape[2]):
                    if np.isnan(datatmax[day, lat,lon]):
                        pass
                    else:
                        b = 30
                        if datatmax[day, lat,lon]<=b:
                           DD[day, lat,lon] = 0
                        elif datatmin[day, lat,lon] < b and datatmax[day, lat,lon]> b:
                           DD[day, lat, lon] = np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-b)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi
                        elif datatmin[day, lat,lon] >=b :
                           DD[day, lat, lon] = (datatmin[day, lat,lon]+datatmax[day, lat,lon])/2-b
        DD = np.nansum(DD, axis = 0)
        for lat in range(0,DD.shape[0]):
            for lon in range(0, DD.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    DD[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1950,0] = year
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(DD)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_BloomKDD30.csv', locals()[str(county)+'_sum'], delimiter = ',')


## harvest eto
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    if year == 1982:
        pass
    else:
        for county in county_list:
            print(county)
            roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
            lat = roitmax.lat.values
            day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
            day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
            roitmax = roitmax.air_temperature[day_start:(day_end+1)]
            shapedata = locals()[str(county)+'_reference'].values[1]
            if roitmax.values.shape[0] == 366:
                fall_start = 213
                fall_end = 304
            else:
                fall_start = 212
                fall_end = 303
            Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
            huss = locals()[str(county)+'huss'+str(year)+'_roi'].specific_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #kg/kg
            rhsmax = locals()[str(county)+'rhsmax'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #%
            rhsmin = locals()[str(county)+'rhsmin'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values
            rsds = (locals()[str(county)+'rsds'+str(year)+'_roi'].surface_downwelling_shortwave_flux_in_air[day_start:(day_end+1)][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
            vpd = locals()[str(county)+'vpd'+str(year)+'_roi'].vpd[day_start:(day_end+1)][fall_start:fall_end+1].values #kPa
            uas = locals()[str(county)+'uas'+str(year)+'_roi'].eastward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values # m/s at 10m
            vas = locals()[str(county)+'vas'+str(year)+'_roi'].northward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values
            roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
            roitmin = roitmin.air_temperature[day_start:(day_end+1)]
            Tmindata = roitmin.values[fall_start:fall_end+1]-273.15
            u_2m = 0.748*((uas**2+vas**2)**0.5)
            Tmean = (Tmaxdata+Tmindata)/2
            D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
            j = np.pi*lat/180
            g = 0.066
            e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
            e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
            ea = (e0_min*rhsmax/100+e0_max*rhsmin/100)/2
            Ra = np.zeros((Tmaxdata.shape[0],Tmaxdata.shape[1],Tmaxdata.shape[2]))
            for day in range(0, Tmaxdata.shape[0]):
                for lon in range(0,Tmaxdata.shape[2]):
                    J = fall_start+day
                    dr = 1+0.033*np.cos(2*np.pi*J/365)
                    d = 0.409*np.sin(2*np.pi*J/365-1.39)
                    ws = np.arccos(-np.tan(j)*np.tan(d))
                    Ra[day,:,lon] = (24*60/np.pi)*0.082*dr*(ws*np.sin(j)*np.sin(d)+np.cos(j)*np.cos(d)*np.sin(ws))
            Rns = 0.77*rsds
            Rso = 0.75*Ra
            Rnl = 4.903e-9*(((Tmaxdata+273.16)**4+(Tmindata+273.16)**4)/2)*(0.34-0.14*(ea**0.5))*(1.35*(rsds/Rso)-0.35)
            Rn = Rns-Rnl
            ETo = (0.408*D*Rn+g*900*u_2m*vpd/(Tmean+273.16))/(D+g*(1+0.34*u_2m))
            ETo_day_num = ETo.shape[0]
            ETo = np.nanmean(ETo,axis = 0)
            for lat in range(0,ETo.shape[0]):
                for lon in range(0, ETo.shape[1]):
                    if np.isnan(shapedata[lat,lon]):
                        ETo[lat,lon] = np.nan
                    locals()[str(county)+'_sum'][year-1950,0] = year
                    locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(ETo)*ETo_day_num
for county in county_list:
    locals()[str(county)+'_sum'][32,0] = 1982
    locals()[str(county)+'_sum'][32,1] = (locals()[str(county)+'_sum'][31,1]+locals()[str(county)+'_sum'][33,1])/2
    savetxt(str(save_path)+str(county)+'_hist_harvest_ETo.csv', locals()[str(county)+'_sum'], delimiter = ',')


## bloom eto
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    if year == 1982:
        pass
    else:
        for county in county_list:
            print(county)
            roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
            lat = roitmax.lat.values
            day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
            day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
            roitmax = roitmax.air_temperature[day_start:(day_end+1)]
            shapedata = locals()[str(county)+'_reference'].values[1]
            if roitmax.values.shape[0] == 366:
                fall_start = 31
                fall_end = 74
            else:
                fall_start = 31
                fall_end = 73
            Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
            huss = locals()[str(county)+'huss'+str(year)+'_roi'].specific_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #kg/kg
            rhsmax = locals()[str(county)+'rhsmax'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #%
            rhsmin = locals()[str(county)+'rhsmin'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values
            rsds = (locals()[str(county)+'rsds'+str(year)+'_roi'].surface_downwelling_shortwave_flux_in_air[day_start:(day_end+1)][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
            vpd = locals()[str(county)+'vpd'+str(year)+'_roi'].vpd[day_start:(day_end+1)][fall_start:fall_end+1].values #kPa
            uas = locals()[str(county)+'uas'+str(year)+'_roi'].eastward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values # m/s at 10m
            vas = locals()[str(county)+'vas'+str(year)+'_roi'].northward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values
            
            roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
            roitmin = roitmin.air_temperature[day_start:(day_end+1)]
            Tmindata = roitmin.values[fall_start:fall_end+1]-273.15
            
            u_2m = 0.748*((uas**2+vas**2)**0.5)
            Tmean = (Tmaxdata+Tmindata)/2
            D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
            j = np.pi*lat/180
            g = 0.066
            e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
            e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
            ea = (e0_min*rhsmax/100+e0_max*rhsmin/100)/2
            Ra = np.zeros((Tmaxdata.shape[0],Tmaxdata.shape[1],Tmaxdata.shape[2]))
            for day in range(0, Tmaxdata.shape[0]):
                for lon in range(0,Tmaxdata.shape[2]):
                    J = fall_start+day
                    dr = 1+0.033*np.cos(2*np.pi*J/365)
                    d = 0.409*np.sin(2*np.pi*J/365-1.39)
                    ws = np.arccos(-np.tan(j)*np.tan(d))
                    Ra[day,:,lon] = (24*60/np.pi)*0.082*dr*(ws*np.sin(j)*np.sin(d)+np.cos(j)*np.cos(d)*np.sin(ws))
            
            Rns = 0.77*rsds
            Rso = 0.75*Ra
            Rnl = 4.903e-9*(((Tmaxdata+273.16)**4+(Tmindata+273.16)**4)/2)*(0.34-0.14*(ea**0.5))*(1.35*(rsds/Rso)-0.35)
            Rn = Rns-Rnl
            ETo = (0.408*D*Rn+g*900*u_2m*vpd/(Tmean+273.16))/(D+g*(1+0.34*u_2m))        
            ETo_day_num = ETo.shape[0]
            ETo = np.nanmean(ETo,axis = 0)
            for lat in range(0,ETo.shape[0]):
                for lon in range(0, ETo.shape[1]):
                    if np.isnan(shapedata[lat,lon]):
                        ETo[lat,lon] = np.nan
                    locals()[str(county)+'_sum'][year-1950,0] = year           
                    locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(ETo)*ETo_day_num
for county in county_list:
    locals()[str(county)+'_sum'][32,0] = 1982
    locals()[str(county)+'_sum'][32,1] = (locals()[str(county)+'_sum'][31,1]+locals()[str(county)+'_sum'][33,1])/2
    savetxt(str(save_path)+str(county)+'_hist_BloomETo.csv', locals()[str(county)+'_sum'], delimiter = ',')


## growing eto
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    if year == 1982:
        pass
    else:
        for county in county_list:
            roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
            lat = roitmax.lat.values
            day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
            day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
            roitmax = roitmax.air_temperature[day_start:(day_end+1)]
            shapedata = locals()[str(county)+'_reference'].values[1]
            if roitmax.values.shape[0] == 366:
                fall_start = 60
                fall_end = 212
            else:
                fall_start = 59
                fall_end = 211
            Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
            huss = locals()[str(county)+'huss'+str(year)+'_roi'].specific_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #kg/kg
            rhsmax = locals()[str(county)+'rhsmax'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #%
            rhsmin = locals()[str(county)+'rhsmin'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values
            rsds = (locals()[str(county)+'rsds'+str(year)+'_roi'].surface_downwelling_shortwave_flux_in_air[day_start:(day_end+1)][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
            vpd = locals()[str(county)+'vpd'+str(year)+'_roi'].vpd[day_start:(day_end+1)][fall_start:fall_end+1].values #kPa
            uas = locals()[str(county)+'uas'+str(year)+'_roi'].eastward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values # m/s at 10m
            vas = locals()[str(county)+'vas'+str(year)+'_roi'].northward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values
            
            roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
            roitmin = roitmin.air_temperature[day_start:(day_end+1)]
            Tmindata = roitmin.values[fall_start:fall_end+1]-273.15
            
            u_2m = 0.748*((uas**2+vas**2)**0.5)
            Tmean = (Tmaxdata+Tmindata)/2
            D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
            j = np.pi*lat/180
            g = 0.066
            e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
            e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
            ea = (e0_min*rhsmax/100+e0_max*rhsmin/100)/2
            Ra = np.zeros((Tmaxdata.shape[0],Tmaxdata.shape[1],Tmaxdata.shape[2]))
            for day in range(0, Tmaxdata.shape[0]):
                for lon in range(0,Tmaxdata.shape[2]):
                    J = fall_start+day
                    dr = 1+0.033*np.cos(2*np.pi*J/365)
                    d = 0.409*np.sin(2*np.pi*J/365-1.39)
                    ws = np.arccos(-np.tan(j)*np.tan(d))
                    Ra[day,:,lon] = (24*60/np.pi)*0.082*dr*(ws*np.sin(j)*np.sin(d)+np.cos(j)*np.cos(d)*np.sin(ws))
            
            Rns = 0.77*rsds
            Rso = 0.75*Ra
            Rnl = 4.903e-9*(((Tmaxdata+273.16)**4+(Tmindata+273.16)**4)/2)*(0.34-0.14*(ea**0.5))*(1.35*(rsds/Rso)-0.35)
            Rn = Rns-Rnl
            ETo = (0.408*D*Rn+g*900*u_2m*vpd/(Tmean+273.16))/(D+g*(1+0.34*u_2m))        
            ETo_day_num = ETo.shape[0]
            ETo = np.nanmean(ETo,axis = 0)
            for lat in range(0,ETo.shape[0]):
                for lon in range(0, ETo.shape[1]):
                    if np.isnan(shapedata[lat,lon]):
                        ETo[lat,lon] = np.nan
                    locals()[str(county)+'_sum'][year-1950,0] = year           
                    locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(ETo)*ETo_day_num
for county in county_list:
    locals()[str(county)+'_sum'][32,0] = 1982
    locals()[str(county)+'_sum'][32,1] = (locals()[str(county)+'_sum'][31,1]+locals()[str(county)+'_sum'][33,1])/2
    savetxt(str(save_path)+str(county)+'_hist_GrowingETo.csv', locals()[str(county)+'_sum'], delimiter = ',')




## Bloom gdd4.5
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        DD = np.zeros((datatmin.shape[0],datatmin.shape[1], datatmin.shape[2]))
        for day in range(0,datatmin.shape[0]):
            for lat in range(0,datatmin.shape[1]):
                for lon in range(0, datatmin.shape[2]):
                    if np.isnan(datatmax[day, lat,lon]):
                        pass
                    else:
                        b = 4.5
                        u = 30
                        if datatmax[day, lat,lon]<=b:
                           DD[day, lat,lon] = 0
                        elif datatmin[day, lat, lon] >=u:
                           DD[day, lat,lon] = 0
                        elif datatmin[day, lat,lon] < b and datatmax[day, lat,lon]> b and datatmax[day, lat,lon] <= u:
                           DD[day, lat, lon] = np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-b)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi
                        elif datatmin[day, lat,lon] < b and datatmax[day, lat,lon]> b and datatmax[day, lat,lon] > u:
                           DD[day, lat, lon] = (np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-b)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi)-(np.arccos((2*u-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-u)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*u-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi)
                        elif datatmin[day, lat,lon] >=b and datatmax[day, lat,lon] <= u:
                           DD[day, lat, lon] = (datatmin[day, lat,lon]+datatmax[day, lat,lon])/2-b
        DD = np.nansum(DD, axis = 0)
        for lat in range(0,DD.shape[0]):
            for lon in range(0, DD.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    DD[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1950,0] = year
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(DD)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_BloomGDD4.csv', locals()[str(county)+'_sum'], delimiter = ',')



## Growing gdd4.5
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        DD = np.zeros((datatmin.shape[0],datatmin.shape[1], datatmin.shape[2]))
        for day in range(0,datatmin.shape[0]):
            for lat in range(0,datatmin.shape[1]):
                for lon in range(0, datatmin.shape[2]):
                    if np.isnan(datatmax[day, lat,lon]):
                        pass
                    else:
                        b = 4.5
                        u = 30
                        if datatmax[day, lat,lon]<=b:
                           DD[day, lat,lon] = 0
                        elif datatmin[day, lat, lon] >=u:
                           DD[day, lat,lon] = 0
                        elif datatmin[day, lat,lon] < b and datatmax[day, lat,lon]> b and datatmax[day, lat,lon] <= u:
                           DD[day, lat, lon] = np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-b)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi
                        elif datatmin[day, lat,lon] < b and datatmax[day, lat,lon]> b and datatmax[day, lat,lon] > u:
                           DD[day, lat, lon] = (np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-b)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi)-(np.arccos((2*u-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-u)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*u-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi)
                        elif datatmin[day, lat,lon] >=b and datatmax[day, lat,lon] <= u:
                           DD[day, lat, lon] = (datatmin[day, lat,lon]+datatmax[day, lat,lon])/2-b
        DD = np.nansum(DD, axis = 0)
        for lat in range(0,DD.shape[0]):
            for lon in range(0, DD.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    DD[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1950,0] = year
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(DD)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_GrowingGDD4.csv', locals()[str(county)+'_sum'], delimiter = ',')





for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((56,2))
for year in range(1950, 2006):
    for county in county_list:
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][-1]
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmin.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1950,0] = year
        locals()[str(county)+'_sum'][year-1950,1] = np.nanmean(Tmindata)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_hist_BloomTmin.csv', locals()[str(county)+'_sum'], delimiter = ',')



period_list = ['2006_2010', '2011_2015', '2016_2020', '2021_2025','2026_2030', '2031_2035', '2036_2040', '2041_2045', '2046_2050', '2051_2055', '2056_2060', '2061_2065', '2066_2070', '2071_2075', '2076_2080', '2081_2085', '2086_2090', '2091_2095', '2096_2099']

cropland_reference = salem.open_xr_dataset('/group/moniergrp/MACA/MACA_mask_no_almond/macav2metdata_tasmin_bcc-csm1-1_r1i1p1_historical_1950_1954_CONUS_daily.nc')
for county in county_list:
    locals()[str(county)+'_reference'] = cropland_reference.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0).salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False).air_temperature
for var in var_list:
    for period in period_list:
        nc_change_lon = nc.Dataset('/group/moniergrp/MACA/macav2metdata_'+str(var)+'_'+str(model_name)+'_r1i1p1_rcp85_'+str(period)+'_CONUS_daily.nc', 'r+')
        if nc_change_lon.variables['lon'][0]+0<-800:
            nc_change_lon.variables['lon'][:] = nc_change_lon.variables['lon'][:]+720
        elif nc_change_lon.variables['lon'][0]+0<-400:
            nc_change_lon.variables['lon'][:] = nc_change_lon.variables['lon'][:]+360
        elif nc_change_lon.variables['lon'][0]+0<-200:
            pass
        elif nc_change_lon.variables['lon'][0]+0>0:
            nc_change_lon.variables['lon'][:] = nc_change_lon.variables['lon'][:]-360
        nc_change_lon.close()
        locals()[str(var)+str(period)] = salem.open_xr_dataset('/group/moniergrp/MACA/macav2metdata_'+str(var)+'_'+str(model_name)+'_r1i1p1_rcp85_'+str(period)+'_CONUS_daily.nc')
        for county in county_list:
            if np.int(np.int(period)/10000) != 2096:
                year_1 = np.int(np.int(period)/10000)
                for year in range(year_1, year_1+5):   
                    locals()[str(county)+str(var)+str(year)+'_roi'] = locals()[str(var)+str(period)].salem.subset(shape = locals()[str(county)+'_shp'], margin = 0).salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
            else:
                year_2 = 2096
                for year in range(year_2, year_2+4):   
                    locals()[str(county)+str(var)+str(year)+'_roi'] = locals()[str(var)+str(period)].salem.subset(shape = locals()[str(county)+'_shp'], margin = 0).salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)

period_list = ['2005_2005']
for var in var_list:
    for period in period_list:
        nc_change_lon = nc.Dataset('/group/moniergrp/MACA/macav2metdata_'+str(var)+'_'+str(model_name)+'_r1i1p1_historical_'+str(period)+'_CONUS_daily.nc', 'r+')
        if nc_change_lon.variables['lon'][0]+0<-800:
            nc_change_lon.variables['lon'][:] = nc_change_lon.variables['lon'][:]+720
        elif nc_change_lon.variables['lon'][0]+0<-400:
            nc_change_lon.variables['lon'][:] = nc_change_lon.variables['lon'][:]+360
        elif nc_change_lon.variables['lon'][0]+0<-200:
            pass
        elif nc_change_lon.variables['lon'][0]+0>0:
            nc_change_lon.variables['lon'][:] = nc_change_lon.variables['lon'][:]-360
        nc_change_lon.close()
        locals()[str(var)+str(period)] = salem.open_xr_dataset('/group/moniergrp/MACA/macav2metdata_'+str(var)+'_'+str(model_name)+'_r1i1p1_historical_'+str(period)+'_CONUS_daily.nc')
        for county in county_list:
            year = 2005
            locals()[str(county)+str(var)+str(year)+'_roi'] = locals()[str(var)+str(period)].salem.subset(shape = locals()[str(county)+'_shp'], margin = 0).salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)


## bloom >12.8
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        sumdays = np.zeros((datatmax.shape[0], datatmax.shape[1], datatmax.shape[2]))
        for day in range(0, datatmax.shape[0]):
            for lat in range(0,datatmax.shape[1]):
                for lon in range(0,datatmax.shape[2]):
                    if (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 > 12.8:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_T12.8.csv', locals()[str(county)+'_sum'], delimiter = ',')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][-1]
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmin.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(Tmindata)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_BloomTmin.csv', locals()[str(county)+'_sum'], delimiter = ',')



##Bloom t15.6
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        sumdays = np.zeros((datatmax.shape[0], datatmax.shape[1], datatmax.shape[2]))
        for day in range(0, datatmax.shape[0]):
            for lat in range(0,datatmax.shape[1]):
                for lon in range(0,datatmax.shape[2]):
                    if (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 > 15.6:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan        
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_T15.6.csv', locals()[str(county)+'_sum'], delimiter = ',')

print('JanFeb >=18.3 <= 26.7')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        sumdays = np.zeros((datatmax.shape[0], datatmax.shape[1], datatmax.shape[2]))
        for day in range(0, datatmax.shape[0]):
            for lat in range(0,datatmax.shape[1]):
                for lon in range(0,datatmax.shape[2]):
                    if (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 >= 18.3 and (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 <= 26.7:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan        
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_T18.3-26.7.csv', locals()[str(county)+'_sum'], delimiter = ',')


print('JanFeb >=10 <= 21.1')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        sumdays = np.zeros((datatmax.shape[0], datatmax.shape[1], datatmax.shape[2]))
        for day in range(0, datatmax.shape[0]):
            for lat in range(0,datatmax.shape[1]):
                for lon in range(0,datatmax.shape[2]):
                    if (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 >= 10 and (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 <= 21.1:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan        
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_T10_21.1.csv', locals()[str(county)+'_sum'], delimiter = ',')



print('JanFeb >=21.1 <= 30.6')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        sumdays = np.zeros((datatmax.shape[0], datatmax.shape[1], datatmax.shape[2]))
        for day in range(0, datatmax.shape[0]):
            for lat in range(0,datatmax.shape[1]):
                for lon in range(0,datatmax.shape[2]):
                    if (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 >= 21.1 and (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 <= 30.6:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan        
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_T21.1-30.6.csv', locals()[str(county)+'_sum'], delimiter = ',')


print('JanFeb >4.4')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        sumdays = np.zeros((datatmax.shape[0], datatmax.shape[1], datatmax.shape[2]))
        for day in range(0, datatmax.shape[0]):
            for lat in range(0,datatmax.shape[1]):
                for lon in range(0,datatmax.shape[2]):
                    if (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 > 4.4:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan        
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_T4.4.csv', locals()[str(county)+'_sum'], delimiter = ',')



    

print('Mar Tmin')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][-1]
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmin.values.shape[0] == 366:
            fall_start = 60
            fall_end = 90
        else:
            fall_start = 59
            fall_end = 89
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(Tmindata)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_MarTmin.csv', locals()[str(county)+'_sum'], delimiter = ',')

##dormancy chill
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata =  locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        roitmax = locals()[str(county)+'tasmax'+str(year-1)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = roitmax.values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        pre_Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year-1)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        pre_Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        WC = np.zeros((Tmindata.shape[0], Tmindata.shape[1],Tmindata.shape[2]))
        pre_WC = np.zeros((pre_Tmindata.shape[0], Tmindata.shape[1],Tmindata.shape[2]))
        for day in range(0, Tmindata.shape[0]):
            for lat in range(0, Tmindata.shape[1]):
                for lon in range(0, Tmindata.shape[2]):
                    if Tmindata[day,lat,lon] >7.22:
                       WC[day,lat,lon] = 0
                    elif Tmindata[day,lat,lon] >0:
                       WC[day,lat,lon] = ((6/((Tmindata[day,lat,lon]+Tmaxdata[day,lat,lon])/2-Tmindata[day,lat,lon]))*(7.22-Tmindata[day,lat,lon]))*2
                    else:
                       WC[day,lat,lon] = ((6/((Tmindata[day,lat,lon]+Tmaxdata[day,lat,lon])/2-Tmindata[day,lat,lon]))*(7.22-Tmindata[day,lat,lon]))*2-((6/((Tmindata[day,lat,lon]+Tmaxdata[day,lat,lon])/2-Tmindata[day,lat,lon]))*(0-Tmindata[day,lat,lon]))*2
        for day in range(0, pre_Tmindata.shape[0]):
            for lat in range(0, Tmindata.shape[1]):
                for lon in range(0, Tmindata.shape[2]):
                    if pre_Tmindata[day,lat,lon] >7.22:
                       pre_WC[day,lat,lon] = 0
                    elif pre_Tmindata[day,lat,lon] >0:
                       pre_WC[day,lat,lon] = ((6/((pre_Tmindata[day,lat,lon]+pre_Tmaxdata[day,lat,lon])/2-pre_Tmindata[day,lat,lon]))*(7.22-pre_Tmindata[day,lat,lon]))*2
                    else:
                       pre_WC[day,lat,lon] = ((6/((pre_Tmindata[day,lat,lon]+pre_Tmaxdata[day,lat,lon])/2-pre_Tmindata[day,lat,lon]))*(7.22-pre_Tmindata[day,lat,lon]))*2-((6/((pre_Tmindata[day,lat,lon]+pre_Tmaxdata[day,lat,lon])/2-pre_Tmindata[day,lat,lon]))*(0-pre_Tmindata[day,lat,lon]))*2

        WC[np.isinf(WC)] = np.nan
        pre_WC[np.isinf(pre_WC)] = np.nan
        WC[np.where(WC>2880)] = np.nan
        pre_WC[np.where(pre_WC>2880)] = np.nan
        WC = np.nansum(WC, axis = 0)
        pre_WC = np.nansum(pre_WC, axis = 0)
        WC = WC + pre_WC
        for lat in range(0,WC.shape[0]):
            for lon in range(0, WC.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    WC[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(WC)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_DormancyChill.csv', locals()[str(county)+'_sum'], delimiter = ',')



#Dormancy Freeze
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        roitmax = locals()[str(county)+'tasmax'+str(year-1)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = roitmax.values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        pre_Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year-1)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        pre_Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        WC = np.zeros((Tmindata.shape[0], Tmindata.shape[1],Tmindata.shape[2]))
        pre_WC = np.zeros((pre_Tmindata.shape[0], Tmindata.shape[1],Tmindata.shape[2]))
        for day in range(0, Tmindata.shape[0]):
            for lat in range(0, Tmindata.shape[1]):
                for lon in range(0, Tmindata.shape[2]):
                    if Tmindata[day,lat,lon] > 0:
                       WC[day,lat,lon] = 0
                    else:
                       WC[day,lat,lon] = ((6/((Tmindata[day,lat,lon]+Tmaxdata[day,lat,lon])/2-Tmindata[day,lat,lon]))*(0-Tmindata[day,lat,lon]))*2
        for day in range(0, pre_Tmindata.shape[0]):
            for lat in range(0, Tmindata.shape[1]):
                for lon in range(0, Tmindata.shape[2]):
                    if pre_Tmindata[day,lat,lon] > 0:
                       pre_WC[day,lat,lon] = 0
                    else:
                       pre_WC[day,lat,lon] = ((6/((pre_Tmindata[day,lat,lon]+pre_Tmaxdata[day,lat,lon])/2-pre_Tmindata[day,lat,lon]))*(0-pre_Tmindata[day,lat,lon]))*2
        WC[np.isinf(WC)] = np.nan
        pre_WC[np.isinf(pre_WC)] = np.nan
        WC[np.where(WC>2880)] = np.nan
        pre_WC[np.where(pre_WC>2880)] = np.nan
        WC = np.nansum(WC, axis = 0)
        pre_WC = np.nansum(pre_WC, axis = 0)
        WC = WC + pre_WC
        for lat in range(0,WC.shape[0]):
            for lon in range(0, WC.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    WC[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(WC)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_DormancyFreeze.csv', locals()[str(county)+'_sum'], delimiter = ',')


##dormancy ETO
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        lat = roitmax.lat.values
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata =  locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15

        huss = locals()[str(county)+'huss'+str(year)+'_roi'].specific_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #kg/kg
        rhsmax = locals()[str(county)+'rhsmax'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #%
        rhsmin = locals()[str(county)+'rhsmin'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values
        rsds =( locals()[str(county)+'rsds'+str(year)+'_roi'].surface_downwelling_shortwave_flux_in_air[day_start:(day_end+1)][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
        vpd = locals()[str(county)+'vpd'+str(year)+'_roi'].vpd[day_start:(day_end+1)][fall_start:fall_end+1].values #kPa
        uas = locals()[str(county)+'uas'+str(year)+'_roi'].eastward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values # m/s at 10m
        vas = locals()[str(county)+'vas'+str(year)+'_roi'].northward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values

        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        u_2m = 0.748*((uas**2+vas**2)**0.5)
        Tmean = (Tmaxdata+Tmindata)/2
        D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
        j = np.pi*lat/180
        g = 0.066
        e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
        e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
        ea = (e0_min*rhsmax/100+e0_max*rhsmin/100)/2
        Ra = np.zeros((Tmaxdata.shape[0],Tmaxdata.shape[1],Tmaxdata.shape[2]))
        for day in range(0, Tmaxdata.shape[0]):
            for lon in range(0,Tmaxdata.shape[2]):
                J = fall_start+day
                dr = 1+0.033*np.cos(2*np.pi*J/365)
                d = 0.409*np.sin(2*np.pi*J/365-1.39)
                ws = np.arccos(-np.tan(j)*np.tan(d))
                Ra[day,:,lon] = (24*60/np.pi)*0.082*dr*(ws*np.sin(j)*np.sin(d)+np.cos(j)*np.cos(d)*np.sin(ws))
        Rns = 0.77*rsds
        Rso = 0.75*Ra
        Rnl = 4.903e-9*(((Tmaxdata+273.16)**4+(Tmindata+273.16)**4)/2)*(0.34-0.14*(ea**0.5))*(1.35*(rsds/Rso)-0.35)
        Rn = Rns-Rnl
        pre_ETo = (0.408*D*Rn+g*900*u_2m*vpd/(Tmean+273.16))/(D+g*(1+0.34*u_2m))
        roitmax = locals()[str(county)+'tasmax'+str(year-1)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = roitmax.values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15

        huss = locals()[str(county)+'huss'+str(year-1)+'_roi'].specific_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #kg/kg
        rhsmax = locals()[str(county)+'rhsmax'+str(year-1)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #%
        rhsmin = locals()[str(county)+'rhsmin'+str(year-1)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values
        rsds = (locals()[str(county)+'rsds'+str(year-1)+'_roi'].surface_downwelling_shortwave_flux_in_air[day_start:(day_end+1)][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
        vpd = locals()[str(county)+'vpd'+str(year-1)+'_roi'].vpd[day_start:(day_end+1)][fall_start:fall_end+1].values #kPa
        uas = locals()[str(county)+'uas'+str(year-1)+'_roi'].eastward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values # m/s at 10m
        vas = locals()[str(county)+'vas'+str(year-1)+'_roi'].northward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values

        roitmin = locals()[str(county)+'tasmin'+str(year-1)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        u_2m = 0.748*((uas**2+vas**2)**0.5)
        Tmean = (Tmaxdata+Tmindata)/2
        D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
        j = np.pi*lat/180
        g = 0.066
        e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
        e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
        ea = (e0_min*rhsmax/100+e0_max*rhsmin/100)/2
        Ra = np.zeros((Tmaxdata.shape[0],Tmaxdata.shape[1],Tmaxdata.shape[2]))
        for day in range(0, Tmaxdata.shape[0]):
            for lon in range(0,Tmaxdata.shape[2]):
                J = fall_start+day
                dr = 1+0.033*np.cos(2*np.pi*J/365)
                d = 0.409*np.sin(2*np.pi*J/365-1.39)
                ws = np.arccos(-np.tan(j)*np.tan(d))
                Ra[day,:,lon] = (24*60/np.pi)*0.082*dr*(ws*np.sin(j)*np.sin(d)+np.cos(j)*np.cos(d)*np.sin(ws))

        Rns = 0.77*rsds
        Rso = 0.75*Ra
        Rnl = 4.903e-9*(((Tmaxdata+273.16)**4+(Tmindata+273.16)**4)/2)*(0.34-0.14*(ea**0.5))*(1.35*(rsds/Rso)-0.35)
        Rn = Rns-Rnl
        ETo = (0.408*D*Rn+g*900*u_2m*vpd/(Tmean+273.16))/(D+g*(1+0.34*u_2m))

        ETo_sum = np.row_stack((pre_ETo,ETo))
        ETo_day_num = ETo_sum.shape[0]
        ETo_sum = np.nanmean(ETo_sum,axis = 0)
        for lat in range(0,ETo_sum.shape[0]):
            for lon in range(0, ETo_sum.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    ETo_sum[lat,lon] = np.nan

        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(ETo_sum)*ETo_day_num
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_DormancyETo.csv', locals()[str(county)+'_sum'], delimiter = ',')







print('Feb ppt')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2)) 
for year in range(2006, 2100):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roi.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roi.time.values).year==year)[0][-1]
        roi = roi.precipitation[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roi.values.shape[0] == 366:
            fall_start = 31
            fall_end = 59
        else:
            fall_start = 31
            fall_end = 58
        data = roi.values[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-2006,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_FebPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')
    
    
print('Feb Tmin')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][-1]
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmin.values.shape[0] == 366:
            fall_start = 31
            fall_end = 59
        else:
            fall_start = 31
            fall_end = 58
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(Tmindata)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_FebTmin.csv', locals()[str(county)+'_sum'], delimiter = ',')
    
    


#bloom sph
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmin = locals()[str(county)+'huss'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][-1]
        roitmin = roitmin.specific_humidity[day_start:(day_end+1)]
        shapedata =  locals()[str(county)+'_reference'].values[1]
        if roitmin.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(Tmindata)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_SpH.csv', locals()[str(county)+'_sum'], delimiter = ',')

for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2)) 
for year in range(2006, 2100):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roi.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roi.time.values).year==year)[0][-1]
        roi = roi.precipitation[day_start:(day_end+1)]
        shapedata = roi.values[1]
        if roi.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        data = roi.values[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-2006,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_JanPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')



##harvest ppt
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roi.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roi.time.values).year==year)[0][-1]
        roi = roi.precipitation[day_start:(day_end+1)]
        shapedata =  locals()[str(county)+'_reference'].values[1]
        if roi.values.shape[0] == 366:
            fall_start = 244
            fall_end = 334
        else:
            fall_start = 243
            fall_end = 333
        data = roi.values[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-2006,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_harvest_Ppt.csv', locals()[str(county)+'_sum'], delimiter = ',')


##harvest tmean
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata =  locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 213
            fall_end = 304
        else:
            fall_start = 212
            fall_end = 303
        datatmax = roitmax.values[fall_start:fall_end+1]
        Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmaxdata[lat,lon] = np.nan
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = (np.nanmean(Tmindata)+np.nanmean(Tmaxdata))/2
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_harvest_Tmean.csv', locals()[str(county)+'_sum'], delimiter = ',')






#Bloomppt
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2)) 
for year in range(2006, 2100):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roi.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roi.time.values).year==year)[0][-1]
        roi = roi.precipitation[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roi.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        data = roi.values[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-2006,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_BloomPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')

## Bloom Tmean
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]
        Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmaxdata[lat,lon] = np.nan
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = (np.nanmean(Tmindata)+np.nanmean(Tmaxdata))/2
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_BloomTmean.csv', locals()[str(county)+'_sum'], delimiter = ',')
    
#Growing Ppt
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2)) 
for year in range(2006, 2100):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roi.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roi.time.values).year==year)[0][-1]
        roi = roi.precipitation[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roi.values.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 60
            fall_end = 211
        data = roi.values[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-2006,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_GrowingPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')
    
## Growing Tmean
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 60
            fall_end = 211
        datatmax = roitmax.values[fall_start:fall_end+1]
        Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmaxdata[lat,lon] = np.nan
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = (np.nanmean(Tmindata)+np.nanmean(Tmaxdata))/2
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_GrowingTmean.csv', locals()[str(county)+'_sum'], delimiter = ',')
    

##dormancy ppt
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year-1)+'_roi']
        day_start = np.where(pd.to_datetime(roi.time.values).year==year-1)[0][0]
        day_end = np.where(pd.to_datetime(roi.time.values).year==year-1)[0][-1]
        roi = roi.precipitation[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roi.values.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        data = roi.values[fall_start:fall_end+1]
        pre_data = np.nansum(data, axis = 0)

        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roi.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roi.time.values).year==year)[0][-1]
        roi = roi.precipitation[day_start:(day_end+1)]
        if roi.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        data = roi.values[fall_start:fall_end+1]
        data = np.nansum(data, axis = 0)
        data = data+pre_data
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_DormancyPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')


## Dormancy Tmean
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata =  locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        datatmax = roitmax.values[fall_start:fall_end+1]
        Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmaxdata[lat,lon] = np.nan
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan

        roitmax = locals()[str(county)+'tasmax'+str(year-1)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = roitmax.values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        datatmax = roitmax.values[fall_start:fall_end+1]
        pre_Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    pre_Tmaxdata[lat,lon] = np.nan
        roitmin = locals()[str(county)+'tasmin'+str(year-1)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]
        pre_Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    pre_Tmindata[lat,lon] = np.nan
        Tmindata = np.row_stack((Tmindata, pre_Tmindata))
        Tmaxdata = np.row_stack((Tmaxdata, pre_Tmaxdata))
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = (np.nanmean(Tmindata)+np.nanmean(Tmaxdata))/2
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_DormancyTmean.csv', locals()[str(county)+'_sum'], delimiter = ',')



##bloom kdd 30
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        DD = np.zeros((datatmin.shape[0],datatmin.shape[1], datatmin.shape[2]))
        for day in range(0,datatmin.shape[0]):
            for lat in range(0,datatmin.shape[1]):
                for lon in range(0, datatmin.shape[2]):
                    if np.isnan(datatmax[day, lat,lon]):
                        pass
                    else:
                        b = 30
                        if datatmax[day, lat,lon]<=b:
                           DD[day, lat,lon] = 0
                        elif datatmin[day, lat,lon] < b and datatmax[day, lat,lon]> b:
                           DD[day, lat, lon] = np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-b)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi
                        elif datatmin[day, lat,lon] >=b :
                           DD[day, lat, lon] = (datatmin[day, lat,lon]+datatmax[day, lat,lon])/2-b
        DD = np.nansum(DD, axis = 0)
        for lat in range(0,DD.shape[0]):
            for lon in range(0, DD.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    DD[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(DD)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_BloomKDD30.csv', locals()[str(county)+'_sum'], delimiter = ',')


## Growing KDD30
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        DD = np.zeros((datatmin.shape[0],datatmin.shape[1], datatmin.shape[2]))
        for day in range(0,datatmin.shape[0]):
            for lat in range(0,datatmin.shape[1]):
                for lon in range(0, datatmin.shape[2]):
                    if np.isnan(datatmax[day, lat,lon]):
                        pass
                    else:
                        b = 30
                        if datatmax[day, lat,lon]<=b:
                           DD[day, lat,lon] = 0
                        elif datatmin[day, lat,lon] < b and datatmax[day, lat,lon]> b:
                           DD[day, lat, lon] = np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-b)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi
                        elif datatmin[day, lat,lon] >=b :
                           DD[day, lat, lon] = (datatmin[day, lat,lon]+datatmax[day, lat,lon])/2-b
        DD = np.nansum(DD, axis = 0)
        for lat in range(0,DD.shape[0]):
            for lon in range(0, DD.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    DD[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(DD)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_GrowingKDD30.csv', locals()[str(county)+'_sum'], delimiter = ',')



## harvest eto
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        lat = roitmax.lat.values
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 213
            fall_end = 304
        else:
            fall_start = 212
            fall_end = 303
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        huss = locals()[str(county)+'huss'+str(year)+'_roi'].specific_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #kg/kg
        rhsmax = locals()[str(county)+'rhsmax'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #%
        rhsmin = locals()[str(county)+'rhsmin'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values
        rsds = (locals()[str(county)+'rsds'+str(year)+'_roi'].surface_downwelling_shortwave_flux_in_air[day_start:(day_end+1)][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
        vpd = locals()[str(county)+'vpd'+str(year)+'_roi'].vpd[day_start:(day_end+1)][fall_start:fall_end+1].values #kPa
        uas = locals()[str(county)+'uas'+str(year)+'_roi'].eastward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values # m/s at 10m
        vas = locals()[str(county)+'vas'+str(year)+'_roi'].northward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values
        
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15
        
        u_2m = 0.748*((uas**2+vas**2)**0.5)
        Tmean = (Tmaxdata+Tmindata)/2
        D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
        j = np.pi*lat/180
        g = 0.066
        e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
        e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
        ea = (e0_min*rhsmax/100+e0_max*rhsmin/100)/2
        Ra = np.zeros((Tmaxdata.shape[0],Tmaxdata.shape[1],Tmaxdata.shape[2]))
        for day in range(0, Tmaxdata.shape[0]):
            for lon in range(0,Tmaxdata.shape[2]):
                J = fall_start+day
                dr = 1+0.033*np.cos(2*np.pi*J/365)
                d = 0.409*np.sin(2*np.pi*J/365-1.39)
                ws = np.arccos(-np.tan(j)*np.tan(d))
                Ra[day,:,lon] = (24*60/np.pi)*0.082*dr*(ws*np.sin(j)*np.sin(d)+np.cos(j)*np.cos(d)*np.sin(ws))
        
        Rns = 0.77*rsds
        Rso = 0.75*Ra
        Rnl = 4.903e-9*(((Tmaxdata+273.16)**4+(Tmindata+273.16)**4)/2)*(0.34-0.14*(ea**0.5))*(1.35*(rsds/Rso)-0.35)
        Rn = Rns-Rnl
        ETo = (0.408*D*Rn+g*900*u_2m*vpd/(Tmean+273.16))/(D+g*(1+0.34*u_2m))        
        ETo_day_num = ETo.shape[0]
        ETo = np.nanmean(ETo,axis = 0)
        for lat in range(0,ETo.shape[0]):
            for lon in range(0, ETo.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    ETo[lat,lon] = np.nan
                locals()[str(county)+'_sum'][year-2006,0] = year           
                locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(ETo)*ETo_day_num
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_harvest_ETo.csv', locals()[str(county)+'_sum'], delimiter = ',')

#bloom  ETo
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        lat = roitmax.lat.values
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        huss = locals()[str(county)+'huss'+str(year)+'_roi'].specific_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #kg/kg
        rhsmax = locals()[str(county)+'rhsmax'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #%
        rhsmin = locals()[str(county)+'rhsmin'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values
        rsds = (locals()[str(county)+'rsds'+str(year)+'_roi'].surface_downwelling_shortwave_flux_in_air[day_start:(day_end+1)][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
        vpd = locals()[str(county)+'vpd'+str(year)+'_roi'].vpd[day_start:(day_end+1)][fall_start:fall_end+1].values #kPa
        uas = locals()[str(county)+'uas'+str(year)+'_roi'].eastward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values # m/s at 10m
        vas = locals()[str(county)+'vas'+str(year)+'_roi'].northward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values
        
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15
        
        u_2m = 0.748*((uas**2+vas**2)**0.5)
        Tmean = (Tmaxdata+Tmindata)/2
        D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
        j = np.pi*lat/180
        g = 0.066
        e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
        e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
        ea = (e0_min*rhsmax/100+e0_max*rhsmin/100)/2
        Ra = np.zeros((Tmaxdata.shape[0],Tmaxdata.shape[1],Tmaxdata.shape[2]))
        for day in range(0, Tmaxdata.shape[0]):
            for lon in range(0,Tmaxdata.shape[2]):
                J = fall_start+day
                dr = 1+0.033*np.cos(2*np.pi*J/365)
                d = 0.409*np.sin(2*np.pi*J/365-1.39)
                ws = np.arccos(-np.tan(j)*np.tan(d))
                Ra[day,:,lon] = (24*60/np.pi)*0.082*dr*(ws*np.sin(j)*np.sin(d)+np.cos(j)*np.cos(d)*np.sin(ws))
        
        Rns = 0.77*rsds
        Rso = 0.75*Ra
        Rnl = 4.903e-9*(((Tmaxdata+273.16)**4+(Tmindata+273.16)**4)/2)*(0.34-0.14*(ea**0.5))*(1.35*(rsds/Rso)-0.35)
        Rn = Rns-Rnl
        ETo = (0.408*D*Rn+g*900*u_2m*vpd/(Tmean+273.16))/(D+g*(1+0.34*u_2m))        
        ETo_day_num = ETo.shape[0]
        ETo = np.nanmean(ETo,axis = 0)
        for lat in range(0,ETo.shape[0]):
            for lon in range(0, ETo.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    ETo[lat,lon] = np.nan
                locals()[str(county)+'_sum'][year-2006,0] = year           
                locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(ETo)*ETo_day_num
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_BloomETo.csv', locals()[str(county)+'_sum'], delimiter = ',')


#growing eto
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        lat = roitmax.lat.values
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        huss = locals()[str(county)+'huss'+str(year)+'_roi'].specific_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #kg/kg
        rhsmax = locals()[str(county)+'rhsmax'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #%
        rhsmin = locals()[str(county)+'rhsmin'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values
        rsds = (locals()[str(county)+'rsds'+str(year)+'_roi'].surface_downwelling_shortwave_flux_in_air[day_start:(day_end+1)][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
        vpd = locals()[str(county)+'vpd'+str(year)+'_roi'].vpd[day_start:(day_end+1)][fall_start:fall_end+1].values #kPa
        uas = locals()[str(county)+'uas'+str(year)+'_roi'].eastward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values # m/s at 10m
        vas = locals()[str(county)+'vas'+str(year)+'_roi'].northward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values
        
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15
        
        u_2m = 0.748*((uas**2+vas**2)**0.5)
        Tmean = (Tmaxdata+Tmindata)/2
        D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
        j = np.pi*lat/180
        g = 0.066
        e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
        e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
        ea = (e0_min*rhsmax/100+e0_max*rhsmin/100)/2
        Ra = np.zeros((Tmaxdata.shape[0],Tmaxdata.shape[1],Tmaxdata.shape[2]))
        for day in range(0, Tmaxdata.shape[0]):
            for lon in range(0,Tmaxdata.shape[2]):
                J = fall_start+day
                dr = 1+0.033*np.cos(2*np.pi*J/365)
                d = 0.409*np.sin(2*np.pi*J/365-1.39)
                ws = np.arccos(-np.tan(j)*np.tan(d))
                Ra[day,:,lon] = (24*60/np.pi)*0.082*dr*(ws*np.sin(j)*np.sin(d)+np.cos(j)*np.cos(d)*np.sin(ws))
        
        Rns = 0.77*rsds
        Rso = 0.75*Ra
        Rnl = 4.903e-9*(((Tmaxdata+273.16)**4+(Tmindata+273.16)**4)/2)*(0.34-0.14*(ea**0.5))*(1.35*(rsds/Rso)-0.35)
        Rn = Rns-Rnl
        ETo = (0.408*D*Rn+g*900*u_2m*vpd/(Tmean+273.16))/(D+g*(1+0.34*u_2m))        
        ETo_day_num = ETo.shape[0]
        ETo = np.nanmean(ETo,axis = 0)
        for lat in range(0,ETo.shape[0]):
            for lon in range(0, ETo.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    ETo[lat,lon] = np.nan
                locals()[str(county)+'_sum'][year-2006,0] = year           
                locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(ETo)*ETo_day_num
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_GrowingETo.csv', locals()[str(county)+'_sum'], delimiter = ',')



## bloom gdd4.5
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        DD = np.zeros((datatmin.shape[0],datatmin.shape[1], datatmin.shape[2]))
        for day in range(0,datatmin.shape[0]):
            for lat in range(0,datatmin.shape[1]):
                for lon in range(0, datatmin.shape[2]):
                    if np.isnan(datatmax[day, lat,lon]):
                        pass
                    else:
                        b = 4.5
                        u = 30
                        if datatmax[day, lat,lon]<=b:
                           DD[day, lat,lon] = 0
                        elif datatmin[day, lat, lon] >=u:
                           DD[day, lat,lon] = 0
                        elif datatmin[day, lat,lon] < b and datatmax[day, lat,lon]> b and datatmax[day, lat,lon] <= u:
                           DD[day, lat, lon] = np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-b)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi
                        elif datatmin[day, lat,lon] < b and datatmax[day, lat,lon]> b and datatmax[day, lat,lon] > u:
                           DD[day, lat, lon] = (np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-b)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi)-(np.arccos((2*u-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-u)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*u-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi)
                        elif datatmin[day, lat,lon] >=b and datatmax[day, lat,lon] <= u:
                           DD[day, lat, lon] = (datatmin[day, lat,lon]+datatmax[day, lat,lon])/2-b
        DD = np.nansum(DD, axis = 0)
        for lat in range(0,DD.shape[0]):
            for lon in range(0, DD.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    DD[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(DD)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_BloomGDD4.csv', locals()[str(county)+'_sum'], delimiter = ',')




##grpwomg gdd4.5
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        DD = np.zeros((datatmin.shape[0],datatmin.shape[1], datatmin.shape[2]))
        for day in range(0,datatmin.shape[0]):
            for lat in range(0,datatmin.shape[1]):
                for lon in range(0, datatmin.shape[2]):
                    if np.isnan(datatmax[day, lat,lon]):
                        pass
                    else:
                        b = 4.5
                        u = 30
                        if datatmax[day, lat,lon]<=b:
                           DD[day, lat,lon] = 0
                        elif datatmin[day, lat, lon] >=u:
                           DD[day, lat,lon] = 0
                        elif datatmin[day, lat,lon] < b and datatmax[day, lat,lon]> b and datatmax[day, lat,lon] <= u:
                           DD[day, lat, lon] = np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-b)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi
                        elif datatmin[day, lat,lon] < b and datatmax[day, lat,lon]> b and datatmax[day, lat,lon] > u:
                           DD[day, lat, lon] = (np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-b)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi)-(np.arccos((2*u-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-u)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*u-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi)
                        elif datatmin[day, lat,lon] >=b and datatmax[day, lat,lon] <= u:
                           DD[day, lat, lon] = (datatmin[day, lat,lon]+datatmax[day, lat,lon])/2-b
        DD = np.nansum(DD, axis = 0)
        for lat in range(0,DD.shape[0]):
            for lon in range(0, DD.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    DD[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(DD)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_GrowingGDD4.csv', locals()[str(county)+'_sum'], delimiter = ',')


##bloom windspeed
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roivas = locals()[str(county)+'vas'+str(year)+'_roi']
        roiuas = locals()[str(county)+'uas'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roivas.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roivas.time.values).year==year)[0][-1]
        roivas = roivas.northward_wind[day_start:(day_end+1)]
        roiuas = roiuas.eastward_wind[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roivas.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        roivas = roivas.values[fall_start:fall_end+1]
        roiuas = roiuas.values[fall_start:fall_end+1]

        sumdays = np.zeros((roivas.shape[0], roivas.shape[1], roivas.shape[2]))
        wpd = (roivas**2+roiuas**2)**0.5
        for day in range(0, roivas.shape[0]):
            for lat in range(0,roivas.shape[1]):
                for lon in range(0,roivas.shape[2]):
                    if wpd[day,lat,lon] > 6.6944:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp85_windspeed.csv', locals()[str(county)+'_sum'], delimiter = ',')


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
import math
county_list = ['Butte', 'Colusa', 'Fresno', 'Glenn', 'Kern', 'Kings', 'Madera', 'Merced', 'San Joaquin', 'Solano', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Yolo', 'Yuba']                      
shapefile = salem.read_shapefile('/home/pgzikala/Shapefiles/CA_Counties/Counties.shp')
for county in county_list:
     locals()[str(county)+'_shp'] = shapefile.loc[shapefile['NAME'].isin([str(county)])]

period_list = ['2006_2010', '2011_2015', '2016_2020', '2021_2025','2026_2030', '2031_2035', '2036_2040', '2041_2045', '2046_2050', '2051_2055', '2056_2060', '2061_2065', '2066_2070', '2071_2075', '2076_2080', '2081_2085', '2086_2090', '2091_2095', '2096_2099']

#model='bcc-csm1-1-m_r1i1p1'
var_list = ['pr', 'tasmin', 'tasmax','huss', 'rhsmax', 'rhsmin', 'rsds', 'vpd', 'uas', 'vas']
cropland_reference = salem.open_xr_dataset('/group/moniergrp/MACA/MACA_mask_no_almond/macav2metdata_tasmin_bcc-csm1-1_r1i1p1_historical_1950_1954_CONUS_daily.nc')
for county in county_list:
    locals()[str(county)+'_reference'] = cropland_reference.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0).salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False).air_temperature
for var in var_list:
    for period in period_list:
        nc_change_lon = nc.Dataset('/group/moniergrp/MACA/macav2metdata_'+str(var)+'_'+str(model_name)+'_r1i1p1_rcp45_'+str(period)+'_CONUS_daily.nc', 'r+')
        if nc_change_lon.variables['lon'][0]+0<-800:
            nc_change_lon.variables['lon'][:] = nc_change_lon.variables['lon'][:]+720
        elif nc_change_lon.variables['lon'][0]+0<-400:
            nc_change_lon.variables['lon'][:] = nc_change_lon.variables['lon'][:]+360
        elif nc_change_lon.variables['lon'][0]+0<-200:
            pass
        elif nc_change_lon.variables['lon'][0]+0>0:
            nc_change_lon.variables['lon'][:] = nc_change_lon.variables['lon'][:]-360
        nc_change_lon.close()
        locals()[str(var)+str(period)] = salem.open_xr_dataset('/group/moniergrp/MACA/macav2metdata_'+str(var)+'_'+str(model_name)+'_r1i1p1_rcp45_'+str(period)+'_CONUS_daily.nc')
        for county in county_list:
            if np.int(np.int(period)/10000) != 2096:
                year_1 = np.int(np.int(period)/10000)
                for year in range(year_1, year_1+5):   
                    locals()[str(county)+str(var)+str(year)+'_roi'] = locals()[str(var)+str(period)].salem.subset(shape = locals()[str(county)+'_shp'], margin = 0).salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
            else:
                year_2 = 2096
                for year in range(year_2, year_2+4):   
                    locals()[str(county)+str(var)+str(year)+'_roi'] = locals()[str(var)+str(period)].salem.subset(shape = locals()[str(county)+'_shp'], margin = 0).salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)

period_list = ['2005_2005']
for var in var_list:
    for period in period_list:
        nc_change_lon = nc.Dataset('/group/moniergrp/MACA/macav2metdata_'+str(var)+'_'+str(model_name)+'_r1i1p1_historical_'+str(period)+'_CONUS_daily.nc', 'r+')
        if nc_change_lon.variables['lon'][0]+0<-800:
            nc_change_lon.variables['lon'][:] = nc_change_lon.variables['lon'][:]+720
        elif nc_change_lon.variables['lon'][0]+0<-400:
            nc_change_lon.variables['lon'][:] = nc_change_lon.variables['lon'][:]+360
        elif nc_change_lon.variables['lon'][0]+0<-200:
            pass
        elif nc_change_lon.variables['lon'][0]+0>0:
            nc_change_lon.variables['lon'][:] = nc_change_lon.variables['lon'][:]-360
        nc_change_lon.close()
        locals()[str(var)+str(period)] = salem.open_xr_dataset('/group/moniergrp/MACA/macav2metdata_'+str(var)+'_'+str(model_name)+'_r1i1p1_historical_'+str(period)+'_CONUS_daily.nc')
        for county in county_list:
            year = 2005
            locals()[str(county)+str(var)+str(year)+'_roi'] = locals()[str(var)+str(period)].salem.subset(shape = locals()[str(county)+'_shp'], margin = 0).salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)


## bloom >12.8
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        sumdays = np.zeros((datatmax.shape[0], datatmax.shape[1], datatmax.shape[2]))
        for day in range(0, datatmax.shape[0]):
            for lat in range(0,datatmax.shape[1]):
                for lon in range(0,datatmax.shape[2]):
                    if (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 > 12.8:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_T12.8.csv', locals()[str(county)+'_sum'], delimiter = ',')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][-1]
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmin.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(Tmindata)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_BloomTmin.csv', locals()[str(county)+'_sum'], delimiter = ',')



##Bloom t15.6
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        sumdays = np.zeros((datatmax.shape[0], datatmax.shape[1], datatmax.shape[2]))
        for day in range(0, datatmax.shape[0]):
            for lat in range(0,datatmax.shape[1]):
                for lon in range(0,datatmax.shape[2]):
                    if (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 > 15.6:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan        
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_T15.6.csv', locals()[str(county)+'_sum'], delimiter = ',')

print('JanFeb >=18.3 <= 26.7')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        sumdays = np.zeros((datatmax.shape[0], datatmax.shape[1], datatmax.shape[2]))
        for day in range(0, datatmax.shape[0]):
            for lat in range(0,datatmax.shape[1]):
                for lon in range(0,datatmax.shape[2]):
                    if (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 >= 18.3 and (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 <= 26.7:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan        
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_T18.3-26.7.csv', locals()[str(county)+'_sum'], delimiter = ',')


print('JanFeb >=10 <= 21.1')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        sumdays = np.zeros((datatmax.shape[0], datatmax.shape[1], datatmax.shape[2]))
        for day in range(0, datatmax.shape[0]):
            for lat in range(0,datatmax.shape[1]):
                for lon in range(0,datatmax.shape[2]):
                    if (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 >= 10 and (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 <= 21.1:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan        
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_T10_21.1.csv', locals()[str(county)+'_sum'], delimiter = ',')



print('JanFeb >=21.1 <= 30.6')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        sumdays = np.zeros((datatmax.shape[0], datatmax.shape[1], datatmax.shape[2]))
        for day in range(0, datatmax.shape[0]):
            for lat in range(0,datatmax.shape[1]):
                for lon in range(0,datatmax.shape[2]):
                    if (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 >= 21.1 and (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 <= 30.6:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan        
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_T21.1-30.6.csv', locals()[str(county)+'_sum'], delimiter = ',')


print('JanFeb >4.4')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        sumdays = np.zeros((datatmax.shape[0], datatmax.shape[1], datatmax.shape[2]))
        for day in range(0, datatmax.shape[0]):
            for lat in range(0,datatmax.shape[1]):
                for lon in range(0,datatmax.shape[2]):
                    if (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 > 4.4:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan        
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_T4.4.csv', locals()[str(county)+'_sum'], delimiter = ',')



    

print('Mar Tmin')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][-1]
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmin.values.shape[0] == 366:
            fall_start = 60
            fall_end = 90
        else:
            fall_start = 59
            fall_end = 89
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(Tmindata)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_MarTmin.csv', locals()[str(county)+'_sum'], delimiter = ',')

##dormancy chill
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata =  locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        roitmax = locals()[str(county)+'tasmax'+str(year-1)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = roitmax.values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        pre_Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year-1)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        pre_Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        WC = np.zeros((Tmindata.shape[0], Tmindata.shape[1],Tmindata.shape[2]))
        pre_WC = np.zeros((pre_Tmindata.shape[0], Tmindata.shape[1],Tmindata.shape[2]))
        for day in range(0, Tmindata.shape[0]):
            for lat in range(0, Tmindata.shape[1]):
                for lon in range(0, Tmindata.shape[2]):
                    if Tmindata[day,lat,lon] >7.22:
                       WC[day,lat,lon] = 0
                    elif Tmindata[day,lat,lon] >0:
                       WC[day,lat,lon] = ((6/((Tmindata[day,lat,lon]+Tmaxdata[day,lat,lon])/2-Tmindata[day,lat,lon]))*(7.22-Tmindata[day,lat,lon]))*2
                    else:
                       WC[day,lat,lon] = ((6/((Tmindata[day,lat,lon]+Tmaxdata[day,lat,lon])/2-Tmindata[day,lat,lon]))*(7.22-Tmindata[day,lat,lon]))*2-((6/((Tmindata[day,lat,lon]+Tmaxdata[day,lat,lon])/2-Tmindata[day,lat,lon]))*(0-Tmindata[day,lat,lon]))*2
        for day in range(0, pre_Tmindata.shape[0]):
            for lat in range(0, Tmindata.shape[1]):
                for lon in range(0, Tmindata.shape[2]):
                    if pre_Tmindata[day,lat,lon] >7.22:
                       pre_WC[day,lat,lon] = 0
                    elif pre_Tmindata[day,lat,lon] >0:
                       pre_WC[day,lat,lon] = ((6/((pre_Tmindata[day,lat,lon]+pre_Tmaxdata[day,lat,lon])/2-pre_Tmindata[day,lat,lon]))*(7.22-pre_Tmindata[day,lat,lon]))*2
                    else:
                       pre_WC[day,lat,lon] = ((6/((pre_Tmindata[day,lat,lon]+pre_Tmaxdata[day,lat,lon])/2-pre_Tmindata[day,lat,lon]))*(7.22-pre_Tmindata[day,lat,lon]))*2-((6/((pre_Tmindata[day,lat,lon]+pre_Tmaxdata[day,lat,lon])/2-pre_Tmindata[day,lat,lon]))*(0-pre_Tmindata[day,lat,lon]))*2

        WC[np.isinf(WC)] = np.nan
        pre_WC[np.isinf(pre_WC)] = np.nan
        WC[np.where(WC>2880)] = np.nan
        pre_WC[np.where(pre_WC>2880)] = np.nan
        WC = np.nansum(WC, axis = 0)
        pre_WC = np.nansum(pre_WC, axis = 0)
        WC = WC + pre_WC
        for lat in range(0,WC.shape[0]):
            for lon in range(0, WC.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    WC[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(WC)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_DormancyChill.csv', locals()[str(county)+'_sum'], delimiter = ',')



#Dormancy Freeze
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        roitmax = locals()[str(county)+'tasmax'+str(year-1)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = roitmax.values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        pre_Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year-1)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        pre_Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        WC = np.zeros((Tmindata.shape[0], Tmindata.shape[1],Tmindata.shape[2]))
        pre_WC = np.zeros((pre_Tmindata.shape[0], Tmindata.shape[1],Tmindata.shape[2]))
        for day in range(0, Tmindata.shape[0]):
            for lat in range(0, Tmindata.shape[1]):
                for lon in range(0, Tmindata.shape[2]):
                    if Tmindata[day,lat,lon] > 0:
                       WC[day,lat,lon] = 0
                    else:
                       WC[day,lat,lon] = ((6/((Tmindata[day,lat,lon]+Tmaxdata[day,lat,lon])/2-Tmindata[day,lat,lon]))*(0-Tmindata[day,lat,lon]))*2
        for day in range(0, pre_Tmindata.shape[0]):
            for lat in range(0, Tmindata.shape[1]):
                for lon in range(0, Tmindata.shape[2]):
                    if pre_Tmindata[day,lat,lon] > 0:
                       pre_WC[day,lat,lon] = 0
                    else:
                       pre_WC[day,lat,lon] = ((6/((pre_Tmindata[day,lat,lon]+pre_Tmaxdata[day,lat,lon])/2-pre_Tmindata[day,lat,lon]))*(0-pre_Tmindata[day,lat,lon]))*2
        WC[np.isinf(WC)] = np.nan
        pre_WC[np.isinf(pre_WC)] = np.nan
        WC[np.where(WC>2880)] = np.nan
        pre_WC[np.where(pre_WC>2880)] = np.nan
        WC = np.nansum(WC, axis = 0)
        pre_WC = np.nansum(pre_WC, axis = 0)
        WC = WC + pre_WC
        for lat in range(0,WC.shape[0]):
            for lon in range(0, WC.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    WC[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(WC)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_DormancyFreeze.csv', locals()[str(county)+'_sum'], delimiter = ',')


##dormancy ETO
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        lat = roitmax.lat.values
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata =  locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15

        huss = locals()[str(county)+'huss'+str(year)+'_roi'].specific_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #kg/kg
        rhsmax = locals()[str(county)+'rhsmax'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #%
        rhsmin = locals()[str(county)+'rhsmin'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values
        rsds =( locals()[str(county)+'rsds'+str(year)+'_roi'].surface_downwelling_shortwave_flux_in_air[day_start:(day_end+1)][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
        vpd = locals()[str(county)+'vpd'+str(year)+'_roi'].vpd[day_start:(day_end+1)][fall_start:fall_end+1].values #kPa
        uas = locals()[str(county)+'uas'+str(year)+'_roi'].eastward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values # m/s at 10m
        vas = locals()[str(county)+'vas'+str(year)+'_roi'].northward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values

        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        u_2m = 0.748*((uas**2+vas**2)**0.5)
        Tmean = (Tmaxdata+Tmindata)/2
        D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
        j = np.pi*lat/180
        g = 0.066
        e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
        e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
        ea = (e0_min*rhsmax/100+e0_max*rhsmin/100)/2
        Ra = np.zeros((Tmaxdata.shape[0],Tmaxdata.shape[1],Tmaxdata.shape[2]))
        for day in range(0, Tmaxdata.shape[0]):
            for lon in range(0,Tmaxdata.shape[2]):
                J = fall_start+day
                dr = 1+0.033*np.cos(2*np.pi*J/365)
                d = 0.409*np.sin(2*np.pi*J/365-1.39)
                ws = np.arccos(-np.tan(j)*np.tan(d))
                Ra[day,:,lon] = (24*60/np.pi)*0.082*dr*(ws*np.sin(j)*np.sin(d)+np.cos(j)*np.cos(d)*np.sin(ws))
        Rns = 0.77*rsds
        Rso = 0.75*Ra
        Rnl = 4.903e-9*(((Tmaxdata+273.16)**4+(Tmindata+273.16)**4)/2)*(0.34-0.14*(ea**0.5))*(1.35*(rsds/Rso)-0.35)
        Rn = Rns-Rnl
        pre_ETo = (0.408*D*Rn+g*900*u_2m*vpd/(Tmean+273.16))/(D+g*(1+0.34*u_2m))
        roitmax = locals()[str(county)+'tasmax'+str(year-1)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = roitmax.values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15

        huss = locals()[str(county)+'huss'+str(year-1)+'_roi'].specific_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #kg/kg
        rhsmax = locals()[str(county)+'rhsmax'+str(year-1)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #%
        rhsmin = locals()[str(county)+'rhsmin'+str(year-1)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values
        rsds = (locals()[str(county)+'rsds'+str(year-1)+'_roi'].surface_downwelling_shortwave_flux_in_air[day_start:(day_end+1)][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
        vpd = locals()[str(county)+'vpd'+str(year-1)+'_roi'].vpd[day_start:(day_end+1)][fall_start:fall_end+1].values #kPa
        uas = locals()[str(county)+'uas'+str(year-1)+'_roi'].eastward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values # m/s at 10m
        vas = locals()[str(county)+'vas'+str(year-1)+'_roi'].northward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values

        roitmin = locals()[str(county)+'tasmin'+str(year-1)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        u_2m = 0.748*((uas**2+vas**2)**0.5)
        Tmean = (Tmaxdata+Tmindata)/2
        D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
        j = np.pi*lat/180
        g = 0.066
        e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
        e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
        ea = (e0_min*rhsmax/100+e0_max*rhsmin/100)/2
        Ra = np.zeros((Tmaxdata.shape[0],Tmaxdata.shape[1],Tmaxdata.shape[2]))
        for day in range(0, Tmaxdata.shape[0]):
            for lon in range(0,Tmaxdata.shape[2]):
                J = fall_start+day
                dr = 1+0.033*np.cos(2*np.pi*J/365)
                d = 0.409*np.sin(2*np.pi*J/365-1.39)
                ws = np.arccos(-np.tan(j)*np.tan(d))
                Ra[day,:,lon] = (24*60/np.pi)*0.082*dr*(ws*np.sin(j)*np.sin(d)+np.cos(j)*np.cos(d)*np.sin(ws))

        Rns = 0.77*rsds
        Rso = 0.75*Ra
        Rnl = 4.903e-9*(((Tmaxdata+273.16)**4+(Tmindata+273.16)**4)/2)*(0.34-0.14*(ea**0.5))*(1.35*(rsds/Rso)-0.35)
        Rn = Rns-Rnl
        ETo = (0.408*D*Rn+g*900*u_2m*vpd/(Tmean+273.16))/(D+g*(1+0.34*u_2m))

        ETo_sum = np.row_stack((pre_ETo,ETo))
        ETo_day_num = ETo_sum.shape[0]
        ETo_sum = np.nanmean(ETo_sum,axis = 0)
        for lat in range(0,ETo_sum.shape[0]):
            for lon in range(0, ETo_sum.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    ETo_sum[lat,lon] = np.nan

        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(ETo_sum)*ETo_day_num
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_DormancyETo.csv', locals()[str(county)+'_sum'], delimiter = ',')







print('Feb ppt')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2)) 
for year in range(2006, 2100):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roi.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roi.time.values).year==year)[0][-1]
        roi = roi.precipitation[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roi.values.shape[0] == 366:
            fall_start = 31
            fall_end = 59
        else:
            fall_start = 31
            fall_end = 58
        data = roi.values[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-2006,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_FebPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')
    
    
print('Feb Tmin')
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][-1]
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmin.values.shape[0] == 366:
            fall_start = 31
            fall_end = 59
        else:
            fall_start = 31
            fall_end = 58
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(Tmindata)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_FebTmin.csv', locals()[str(county)+'_sum'], delimiter = ',')
    
    


#bloom sph
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmin = locals()[str(county)+'huss'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmin.time.values).year==year)[0][-1]
        roitmin = roitmin.specific_humidity[day_start:(day_end+1)]
        shapedata =  locals()[str(county)+'_reference'].values[1]
        if roitmin.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(Tmindata)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_SpH.csv', locals()[str(county)+'_sum'], delimiter = ',')

for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2)) 
for year in range(2006, 2100):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roi.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roi.time.values).year==year)[0][-1]
        roi = roi.precipitation[day_start:(day_end+1)]
        shapedata = roi.values[1]
        if roi.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        data = roi.values[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-2006,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_JanPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')



##harvest ppt
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roi.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roi.time.values).year==year)[0][-1]
        roi = roi.precipitation[day_start:(day_end+1)]
        shapedata =  locals()[str(county)+'_reference'].values[1]
        if roi.values.shape[0] == 366:
            fall_start = 244
            fall_end = 334
        else:
            fall_start = 243
            fall_end = 333
        data = roi.values[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-2006,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_harvest_Ppt.csv', locals()[str(county)+'_sum'], delimiter = ',')


##harvest tmean
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata =  locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 213
            fall_end = 304
        else:
            fall_start = 212
            fall_end = 303
        datatmax = roitmax.values[fall_start:fall_end+1]
        Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmaxdata[lat,lon] = np.nan
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = (np.nanmean(Tmindata)+np.nanmean(Tmaxdata))/2
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_harvest_Tmean.csv', locals()[str(county)+'_sum'], delimiter = ',')






#Bloomppt
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2)) 
for year in range(2006, 2100):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roi.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roi.time.values).year==year)[0][-1]
        roi = roi.precipitation[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roi.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        data = roi.values[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-2006,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_BloomPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')

## Bloom Tmean
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]
        Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmaxdata[lat,lon] = np.nan
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = (np.nanmean(Tmindata)+np.nanmean(Tmaxdata))/2
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_BloomTmean.csv', locals()[str(county)+'_sum'], delimiter = ',')
    
#Growing Ppt
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2)) 
for year in range(2006, 2100):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roi.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roi.time.values).year==year)[0][-1]
        roi = roi.precipitation[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roi.values.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 60
            fall_end = 211
        data = roi.values[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-2006,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_GrowingPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')
    
## Growing Tmean
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 60
            fall_end = 211
        datatmax = roitmax.values[fall_start:fall_end+1]
        Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmaxdata[lat,lon] = np.nan
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-2006,0] = year           
        locals()[str(county)+'_sum'][year-2006,1] = (np.nanmean(Tmindata)+np.nanmean(Tmaxdata))/2
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_GrowingTmean.csv', locals()[str(county)+'_sum'], delimiter = ',')
    

##dormancy ppt
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year-1)+'_roi']
        day_start = np.where(pd.to_datetime(roi.time.values).year==year-1)[0][0]
        day_end = np.where(pd.to_datetime(roi.time.values).year==year-1)[0][-1]
        roi = roi.precipitation[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roi.values.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        data = roi.values[fall_start:fall_end+1]
        pre_data = np.nansum(data, axis = 0)

        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roi.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roi.time.values).year==year)[0][-1]
        roi = roi.precipitation[day_start:(day_end+1)]
        if roi.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        data = roi.values[fall_start:fall_end+1]
        data = np.nansum(data, axis = 0)
        data = data+pre_data
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_DormancyPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')


## Dormancy Tmean
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata =  locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        datatmax = roitmax.values[fall_start:fall_end+1]
        Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmaxdata[lat,lon] = np.nan
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan

        roitmax = locals()[str(county)+'tasmax'+str(year-1)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year-1)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = roitmax.values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        datatmax = roitmax.values[fall_start:fall_end+1]
        pre_Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    pre_Tmaxdata[lat,lon] = np.nan
        roitmin = locals()[str(county)+'tasmin'+str(year-1)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]
        pre_Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    pre_Tmindata[lat,lon] = np.nan
        Tmindata = np.row_stack((Tmindata, pre_Tmindata))
        Tmaxdata = np.row_stack((Tmaxdata, pre_Tmaxdata))
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = (np.nanmean(Tmindata)+np.nanmean(Tmaxdata))/2
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_DormancyTmean.csv', locals()[str(county)+'_sum'], delimiter = ',')



##bloom kdd 30
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        DD = np.zeros((datatmin.shape[0],datatmin.shape[1], datatmin.shape[2]))
        for day in range(0,datatmin.shape[0]):
            for lat in range(0,datatmin.shape[1]):
                for lon in range(0, datatmin.shape[2]):
                    if np.isnan(datatmax[day, lat,lon]):
                        pass
                    else:
                        b = 30
                        if datatmax[day, lat,lon]<=b:
                           DD[day, lat,lon] = 0
                        elif datatmin[day, lat,lon] < b and datatmax[day, lat,lon]> b:
                           DD[day, lat, lon] = np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-b)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi
                        elif datatmin[day, lat,lon] >=b :
                           DD[day, lat, lon] = (datatmin[day, lat,lon]+datatmax[day, lat,lon])/2-b
        DD = np.nansum(DD, axis = 0)
        for lat in range(0,DD.shape[0]):
            for lon in range(0, DD.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    DD[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(DD)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_BloomKDD30.csv', locals()[str(county)+'_sum'], delimiter = ',')


## Growing KDD30
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        DD = np.zeros((datatmin.shape[0],datatmin.shape[1], datatmin.shape[2]))
        for day in range(0,datatmin.shape[0]):
            for lat in range(0,datatmin.shape[1]):
                for lon in range(0, datatmin.shape[2]):
                    if np.isnan(datatmax[day, lat,lon]):
                        pass
                    else:
                        b = 30
                        if datatmax[day, lat,lon]<=b:
                           DD[day, lat,lon] = 0
                        elif datatmin[day, lat,lon] < b and datatmax[day, lat,lon]> b:
                           DD[day, lat, lon] = np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-b)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi
                        elif datatmin[day, lat,lon] >=b :
                           DD[day, lat, lon] = (datatmin[day, lat,lon]+datatmax[day, lat,lon])/2-b
        DD = np.nansum(DD, axis = 0)
        for lat in range(0,DD.shape[0]):
            for lon in range(0, DD.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    DD[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(DD)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_GrowingKDD30.csv', locals()[str(county)+'_sum'], delimiter = ',')



## harvest eto
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        lat = roitmax.lat.values
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 213
            fall_end = 304
        else:
            fall_start = 212
            fall_end = 303
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        huss = locals()[str(county)+'huss'+str(year)+'_roi'].specific_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #kg/kg
        rhsmax = locals()[str(county)+'rhsmax'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #%
        rhsmin = locals()[str(county)+'rhsmin'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values
        rsds = (locals()[str(county)+'rsds'+str(year)+'_roi'].surface_downwelling_shortwave_flux_in_air[day_start:(day_end+1)][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
        vpd = locals()[str(county)+'vpd'+str(year)+'_roi'].vpd[day_start:(day_end+1)][fall_start:fall_end+1].values #kPa
        uas = locals()[str(county)+'uas'+str(year)+'_roi'].eastward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values # m/s at 10m
        vas = locals()[str(county)+'vas'+str(year)+'_roi'].northward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values
        
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15
        
        u_2m = 0.748*((uas**2+vas**2)**0.5)
        Tmean = (Tmaxdata+Tmindata)/2
        D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
        j = np.pi*lat/180
        g = 0.066
        e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
        e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
        ea = (e0_min*rhsmax/100+e0_max*rhsmin/100)/2
        Ra = np.zeros((Tmaxdata.shape[0],Tmaxdata.shape[1],Tmaxdata.shape[2]))
        for day in range(0, Tmaxdata.shape[0]):
            for lon in range(0,Tmaxdata.shape[2]):
                J = fall_start+day
                dr = 1+0.033*np.cos(2*np.pi*J/365)
                d = 0.409*np.sin(2*np.pi*J/365-1.39)
                ws = np.arccos(-np.tan(j)*np.tan(d))
                Ra[day,:,lon] = (24*60/np.pi)*0.082*dr*(ws*np.sin(j)*np.sin(d)+np.cos(j)*np.cos(d)*np.sin(ws))
        
        Rns = 0.77*rsds
        Rso = 0.75*Ra
        Rnl = 4.903e-9*(((Tmaxdata+273.16)**4+(Tmindata+273.16)**4)/2)*(0.34-0.14*(ea**0.5))*(1.35*(rsds/Rso)-0.35)
        Rn = Rns-Rnl
        ETo = (0.408*D*Rn+g*900*u_2m*vpd/(Tmean+273.16))/(D+g*(1+0.34*u_2m))        
        ETo_day_num = ETo.shape[0]
        ETo = np.nanmean(ETo,axis = 0)
        for lat in range(0,ETo.shape[0]):
            for lon in range(0, ETo.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    ETo[lat,lon] = np.nan
                locals()[str(county)+'_sum'][year-2006,0] = year           
                locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(ETo)*ETo_day_num
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_harvest_ETo.csv', locals()[str(county)+'_sum'], delimiter = ',')

#bloom  ETo
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        lat = roitmax.lat.values
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        huss = locals()[str(county)+'huss'+str(year)+'_roi'].specific_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #kg/kg
        rhsmax = locals()[str(county)+'rhsmax'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #%
        rhsmin = locals()[str(county)+'rhsmin'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values
        rsds = (locals()[str(county)+'rsds'+str(year)+'_roi'].surface_downwelling_shortwave_flux_in_air[day_start:(day_end+1)][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
        vpd = locals()[str(county)+'vpd'+str(year)+'_roi'].vpd[day_start:(day_end+1)][fall_start:fall_end+1].values #kPa
        uas = locals()[str(county)+'uas'+str(year)+'_roi'].eastward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values # m/s at 10m
        vas = locals()[str(county)+'vas'+str(year)+'_roi'].northward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values
        
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15
        
        u_2m = 0.748*((uas**2+vas**2)**0.5)
        Tmean = (Tmaxdata+Tmindata)/2
        D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
        j = np.pi*lat/180
        g = 0.066
        e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
        e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
        ea = (e0_min*rhsmax/100+e0_max*rhsmin/100)/2
        Ra = np.zeros((Tmaxdata.shape[0],Tmaxdata.shape[1],Tmaxdata.shape[2]))
        for day in range(0, Tmaxdata.shape[0]):
            for lon in range(0,Tmaxdata.shape[2]):
                J = fall_start+day
                dr = 1+0.033*np.cos(2*np.pi*J/365)
                d = 0.409*np.sin(2*np.pi*J/365-1.39)
                ws = np.arccos(-np.tan(j)*np.tan(d))
                Ra[day,:,lon] = (24*60/np.pi)*0.082*dr*(ws*np.sin(j)*np.sin(d)+np.cos(j)*np.cos(d)*np.sin(ws))
        
        Rns = 0.77*rsds
        Rso = 0.75*Ra
        Rnl = 4.903e-9*(((Tmaxdata+273.16)**4+(Tmindata+273.16)**4)/2)*(0.34-0.14*(ea**0.5))*(1.35*(rsds/Rso)-0.35)
        Rn = Rns-Rnl
        ETo = (0.408*D*Rn+g*900*u_2m*vpd/(Tmean+273.16))/(D+g*(1+0.34*u_2m))        
        ETo_day_num = ETo.shape[0]
        ETo = np.nanmean(ETo,axis = 0)
        for lat in range(0,ETo.shape[0]):
            for lon in range(0, ETo.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    ETo[lat,lon] = np.nan
                locals()[str(county)+'_sum'][year-2006,0] = year           
                locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(ETo)*ETo_day_num
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_BloomETo.csv', locals()[str(county)+'_sum'], delimiter = ',')


#growing eto
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        lat = roitmax.lat.values
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        huss = locals()[str(county)+'huss'+str(year)+'_roi'].specific_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #kg/kg
        rhsmax = locals()[str(county)+'rhsmax'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values #%
        rhsmin = locals()[str(county)+'rhsmin'+str(year)+'_roi'].relative_humidity[day_start:(day_end+1)][fall_start:fall_end+1].values
        rsds = (locals()[str(county)+'rsds'+str(year)+'_roi'].surface_downwelling_shortwave_flux_in_air[day_start:(day_end+1)][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
        vpd = locals()[str(county)+'vpd'+str(year)+'_roi'].vpd[day_start:(day_end+1)][fall_start:fall_end+1].values #kPa
        uas = locals()[str(county)+'uas'+str(year)+'_roi'].eastward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values # m/s at 10m
        vas = locals()[str(county)+'vas'+str(year)+'_roi'].northward_wind[day_start:(day_end+1)][fall_start:fall_end+1].values
        
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15
        
        u_2m = 0.748*((uas**2+vas**2)**0.5)
        Tmean = (Tmaxdata+Tmindata)/2
        D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
        j = np.pi*lat/180
        g = 0.066
        e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
        e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
        ea = (e0_min*rhsmax/100+e0_max*rhsmin/100)/2
        Ra = np.zeros((Tmaxdata.shape[0],Tmaxdata.shape[1],Tmaxdata.shape[2]))
        for day in range(0, Tmaxdata.shape[0]):
            for lon in range(0,Tmaxdata.shape[2]):
                J = fall_start+day
                dr = 1+0.033*np.cos(2*np.pi*J/365)
                d = 0.409*np.sin(2*np.pi*J/365-1.39)
                ws = np.arccos(-np.tan(j)*np.tan(d))
                Ra[day,:,lon] = (24*60/np.pi)*0.082*dr*(ws*np.sin(j)*np.sin(d)+np.cos(j)*np.cos(d)*np.sin(ws))
        
        Rns = 0.77*rsds
        Rso = 0.75*Ra
        Rnl = 4.903e-9*(((Tmaxdata+273.16)**4+(Tmindata+273.16)**4)/2)*(0.34-0.14*(ea**0.5))*(1.35*(rsds/Rso)-0.35)
        Rn = Rns-Rnl
        ETo = (0.408*D*Rn+g*900*u_2m*vpd/(Tmean+273.16))/(D+g*(1+0.34*u_2m))        
        ETo_day_num = ETo.shape[0]
        ETo = np.nanmean(ETo,axis = 0)
        for lat in range(0,ETo.shape[0]):
            for lon in range(0, ETo.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    ETo[lat,lon] = np.nan
                locals()[str(county)+'_sum'][year-2006,0] = year           
                locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(ETo)*ETo_day_num
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_GrowingETo.csv', locals()[str(county)+'_sum'], delimiter = ',')



## bloom gdd4.5
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        DD = np.zeros((datatmin.shape[0],datatmin.shape[1], datatmin.shape[2]))
        for day in range(0,datatmin.shape[0]):
            for lat in range(0,datatmin.shape[1]):
                for lon in range(0, datatmin.shape[2]):
                    if np.isnan(datatmax[day, lat,lon]):
                        pass
                    else:
                        b = 4.5
                        u = 30
                        if datatmax[day, lat,lon]<=b:
                           DD[day, lat,lon] = 0
                        elif datatmin[day, lat, lon] >=u:
                           DD[day, lat,lon] = 0
                        elif datatmin[day, lat,lon] < b and datatmax[day, lat,lon]> b and datatmax[day, lat,lon] <= u:
                           DD[day, lat, lon] = np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-b)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi
                        elif datatmin[day, lat,lon] < b and datatmax[day, lat,lon]> b and datatmax[day, lat,lon] > u:
                           DD[day, lat, lon] = (np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-b)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi)-(np.arccos((2*u-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-u)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*u-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi)
                        elif datatmin[day, lat,lon] >=b and datatmax[day, lat,lon] <= u:
                           DD[day, lat, lon] = (datatmin[day, lat,lon]+datatmax[day, lat,lon])/2-b
        DD = np.nansum(DD, axis = 0)
        for lat in range(0,DD.shape[0]):
            for lon in range(0, DD.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    DD[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(DD)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_BloomGDD4.csv', locals()[str(county)+'_sum'], delimiter = ',')




##grpwomg gdd4.5
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roitmax.time.values).year==year)[0][-1]
        roitmax = roitmax.air_temperature[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roitmax.values.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        roitmin = roitmin.air_temperature[day_start:(day_end+1)]
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        DD = np.zeros((datatmin.shape[0],datatmin.shape[1], datatmin.shape[2]))
        for day in range(0,datatmin.shape[0]):
            for lat in range(0,datatmin.shape[1]):
                for lon in range(0, datatmin.shape[2]):
                    if np.isnan(datatmax[day, lat,lon]):
                        pass
                    else:
                        b = 4.5
                        u = 30
                        if datatmax[day, lat,lon]<=b:
                           DD[day, lat,lon] = 0
                        elif datatmin[day, lat, lon] >=u:
                           DD[day, lat,lon] = 0
                        elif datatmin[day, lat,lon] < b and datatmax[day, lat,lon]> b and datatmax[day, lat,lon] <= u:
                           DD[day, lat, lon] = np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-b)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi
                        elif datatmin[day, lat,lon] < b and datatmax[day, lat,lon]> b and datatmax[day, lat,lon] > u:
                           DD[day, lat, lon] = (np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-b)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*b-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi)-(np.arccos((2*u-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon]))*((datatmax[day, lat,lon]+datatmin[day, lat,lon])/2-u)/math.pi+(datatmax[day, lat,lon]-datatmin[day, lat,lon])*np.sin(np.arccos((2*u-datatmax[day, lat,lon]-datatmin[day, lat,lon])/(datatmax[day, lat,lon]-datatmin[day, lat,lon])))/2/math.pi)
                        elif datatmin[day, lat,lon] >=b and datatmax[day, lat,lon] <= u:
                           DD[day, lat, lon] = (datatmin[day, lat,lon]+datatmax[day, lat,lon])/2-b
        DD = np.nansum(DD, axis = 0)
        for lat in range(0,DD.shape[0]):
            for lon in range(0, DD.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    DD[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(DD)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_GrowingGDD4.csv', locals()[str(county)+'_sum'], delimiter = ',')


##bloom windspeed
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((94,2))
for year in range(2006, 2100):
    for county in county_list:
        roivas = locals()[str(county)+'vas'+str(year)+'_roi']
        roiuas = locals()[str(county)+'uas'+str(year)+'_roi']
        day_start = np.where(pd.to_datetime(roivas.time.values).year==year)[0][0]
        day_end = np.where(pd.to_datetime(roivas.time.values).year==year)[0][-1]
        roivas = roivas.northward_wind[day_start:(day_end+1)]
        roiuas = roiuas.eastward_wind[day_start:(day_end+1)]
        shapedata = locals()[str(county)+'_reference'].values[1]
        if roivas.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        roivas = roivas.values[fall_start:fall_end+1]
        roiuas = roiuas.values[fall_start:fall_end+1]

        sumdays = np.zeros((roivas.shape[0], roivas.shape[1], roivas.shape[2]))
        wpd = (roivas**2+roiuas**2)**0.5
        for day in range(0, roivas.shape[0]):
            for lat in range(0,roivas.shape[1]):
                for lon in range(0,roivas.shape[2]):
                    if wpd[day,lat,lon] > 6.6944:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2006,0] = year
        locals()[str(county)+'_sum'][year-2006,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_rcp45_windspeed.csv', locals()[str(county)+'_sum'], delimiter = ',')


