##Extract climate variables from gridMET and calculate ACIs for each county

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
import math

data_ID='11_19'  ## set ID for the entire simulation
save_path ='/home/shqwu/Almond_code_git/saved_data/'+str(data_ID)+'/Gridmet_ACI/'

county_list = ['Butte', 'Colusa', 'Fresno', 'Glenn', 'Kern', 'Kings', 'Madera', 'Merced', 'San Joaquin', 'Solano', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Yolo', 'Yuba']                      
shapefile = salem.read_shapefile('/home/shqwu/Almond_code_git/CA_Counties/Counties.shp')
for county in county_list:
     locals()[str(county)+'_shp'] = shapefile.loc[shapefile['NAME'].isin([str(county)])]

#harvest ETo
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))    
for year in range(1980, 2021):
    ncdata = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.ETo._'+str(year)+'.nc')
    for county in county_list:
        subset = ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi.potential_evapotranspiration[1]
        if roi.potential_evapotranspiration.shape[0] == 366:
            fall_start = 213 
            fall_end = 304
        else:
            fall_start = 212
            fall_end = 303
        data = roi.potential_evapotranspiration[fall_start:fall_end+1].values
        locals()[str(county)+'_sum'][year-1980,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_harvest_ETo.csv', locals()[str(county)+'_sum'], delimiter = ',')
#harvest ppt
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdata = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Precip.'+str(year)+'.nc')
    for county in county_list:
        subset = ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi.precipitation_amount[1]
        if roi.precipitation_amount.shape[0] == 366:
            fall_start = 213
            fall_end = 304
        else:
            fall_start = 212
            fall_end = 303
        data = roi.precipitation_amount[fall_start:fall_end+1].values
        locals()[str(county)+'_sum'][year-1980,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_harvest_Ppt.csv', locals()[str(county)+'_sum'], delimiter = ',')


#Jan ppt
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdata = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Precip.'+str(year)+'.nc')
    for county in county_list:
        subset = ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi.precipitation_amount[1]
        if roi.precipitation_amount.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        data = roi.precipitation_amount[fall_start:fall_end+1].values
        locals()[str(county)+'_sum'][year-1980,0] = year
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_JanPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')



## harvest Tmean
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmax.'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year)+'.nc')
    for county in county_list:
        subsettmax = ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax.air_temperature[1]
        if roitmax.air_temperature.shape[0] == 366:
            fall_start = 214
            fall_end = 304
        else:
            fall_start = 213
            fall_end = 303
        datatmax = roitmax.air_temperature[fall_start:fall_end+1].values
        Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmaxdata[lat,lon] = np.nan
        subsettmin = ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)                    
        datatmin = roitmin.air_temperature[fall_start:fall_end+1].values
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-1980,0] = year           
        locals()[str(county)+'_sum'][year-1980,1] = (np.nanmean(Tmindata)+np.nanmean(Tmaxdata))/2
for county in county_list:
    savetxt(str(save_path)+str(county)+'_harvest_Tmean.csv', locals()[str(county)+'_sum'], delimiter = ',')



#Bloom Ppt
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdata = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Precip.'+str(year)+'.nc')
    for county in county_list:
        subset = ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi.precipitation_amount[1]
        if roi.precipitation_amount.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        data = roi.precipitation_amount[fall_start:fall_end+1].values
        locals()[str(county)+'_sum'][year-1980,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_BloomPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')
    
    
## Bloom Tmean
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmax.'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year)+'.nc')
    for county in county_list:
        subsettmax = ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax.air_temperature[1]
        if roitmax.air_temperature.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.air_temperature[fall_start:fall_end+1].values
        Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmaxdata[lat,lon] = np.nan
        subsettmin = ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)                    
        datatmin = roitmin.air_temperature[fall_start:fall_end+1].values
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-1980,0] = year           
        locals()[str(county)+'_sum'][year-1980,1] = (np.nanmean(Tmindata)+np.nanmean(Tmaxdata))/2
for county in county_list:
    savetxt(str(save_path)+str(county)+'_BloomTmean.csv', locals()[str(county)+'_sum'], delimiter = ',')
    

#Growing Precp
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdata = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Precip.'+str(year)+'.nc')
    for county in county_list:
        subset = ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi.precipitation_amount[1]
        if roi.precipitation_amount.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        data = roi.precipitation_amount[fall_start:fall_end+1].values
        locals()[str(county)+'_sum'][year-1980,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_GrowingPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')
    
## Growing Tmean
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmax.'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year)+'.nc')
    for county in county_list:
        subsettmax = ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax.air_temperature[1]
        if roitmax.air_temperature.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        datatmax = roitmax.air_temperature[fall_start:fall_end+1].values
        Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmaxdata[lat,lon] = np.nan
        subsettmin = ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)                    
        datatmin = roitmin.air_temperature[fall_start:fall_end+1].values
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-1980,0] = year           
        locals()[str(county)+'_sum'][year-1980,1] = (np.nanmean(Tmindata)+np.nanmean(Tmaxdata))/2
for county in county_list:
    savetxt(str(save_path)+str(county)+'_GrowingTmean.csv', locals()[str(county)+'_sum'], delimiter = ',')  


##Dormancy precp
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    pre_ncdata = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Precip.'+str(year-1)+'.nc')
    ncdata = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Precip.'+str(year)+'.nc')
    for county in county_list:
        subset = ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi.precipitation_amount[1]
        if roi.precipitation_amount.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        data = roi.precipitation_amount[fall_start:fall_end+1].values
        pre_data = np.nansum(data, axis = 0)
        
        subset = pre_ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi.precipitation_amount[1]
        if roi.precipitation_amount.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        data = roi.precipitation_amount[fall_start:fall_end+1].values
        data = np.nansum(data, axis = 0)        
        data = data+pre_data
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1980,0] = year   
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_DormancyPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')

##Dormancy ETo
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    pre_ncdata = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.ETo._'+str(year-1)+'.nc')
    ncdata = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.ETo._'+str(year)+'.nc')
    for county in county_list:
        subset = ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi.potential_evapotranspiration[1]
        if roi.potential_evapotranspiration.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        data = roi.potential_evapotranspiration[fall_start:fall_end+1].values
        pre_data = np.nansum(data, axis = 0)        
        subset = pre_ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi.potential_evapotranspiration[1]
        if roi.potential_evapotranspiration.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        data = roi.potential_evapotranspiration[fall_start:fall_end+1].values
        data = np.nansum(data, axis = 0)        
        data = data+pre_data
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1980,0] = year   
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_DormancyETo.csv', locals()[str(county)+'_sum'], delimiter = ',')


## Dormancy Tmean
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmax.'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year)+'.nc')
    pre_ncdatatmax = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmax.'+str(year-1)+'.nc')
    pre_ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year-1)+'.nc')
    for county in county_list:
        subsettmax = ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax.air_temperature[1]
        if roitmax.air_temperature.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        datatmax = roitmax.air_temperature[fall_start:fall_end+1].values
        Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmaxdata[lat,lon] = np.nan
        subsettmin = ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)                    
        datatmin = roitmin.air_temperature[fall_start:fall_end+1].values
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan
        
        subsettmax = pre_ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax.air_temperature[1]
        if roitmax.air_temperature.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        datatmax = roitmax.air_temperature[fall_start:fall_end+1].values
        pre_Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    pre_Tmaxdata[lat,lon] = np.nan
        subsettmin = pre_ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)                    
        datatmin = roitmin.air_temperature[fall_start:fall_end+1].values
        pre_Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    pre_Tmindata[lat,lon] = np.nan        
        Tmindata = np.row_stack((Tmindata, pre_Tmindata))
        Tmaxdata = np.row_stack((Tmaxdata, pre_Tmaxdata))
        locals()[str(county)+'_sum'][year-1980,0] = year           
        locals()[str(county)+'_sum'][year-1980,1] = (np.nanmean(Tmindata)+np.nanmean(Tmaxdata))/2
for county in county_list:
    savetxt(str(save_path)+str(county)+'_DormancyTmean.csv', locals()[str(county)+'_sum'], delimiter = ',')
    




##GrowingKDD 30

for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmax.'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year)+'.nc')
    for county in county_list:
        subsettmax = ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax.air_temperature[1]
        if roitmax.air_temperature.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        datatmax = roitmax.air_temperature[fall_start:fall_end+1].values-273.15
        subsettmin = ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        datatmin = roitmin.air_temperature[fall_start:fall_end+1].values-273.15
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
        locals()[str(county)+'_sum'][year-1980,0] = year
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(DD)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_GrowingKDD30.csv', locals()[str(county)+'_sum'], delimiter = ',')


##BloomKDD 30

for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmax.'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year)+'.nc')
    for county in county_list:
        subsettmax = ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax.air_temperature[1]
        if roitmax.air_temperature.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.air_temperature[fall_start:fall_end+1].values-273.15
        subsettmin = ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        datatmin = roitmin.air_temperature[fall_start:fall_end+1].values-273.15
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
        locals()[str(county)+'_sum'][year-1980,0] = year
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(DD)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_BloomKDD30.csv', locals()[str(county)+'_sum'], delimiter = ',')



##BloomGDD4.5

for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmax.'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year)+'.nc')
    for county in county_list:
        subsettmax = ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax.air_temperature[1]
        if roitmax.air_temperature.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.air_temperature[fall_start:fall_end+1].values-273.15
        subsettmin = ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        datatmin = roitmin.air_temperature[fall_start:fall_end+1].values-273.15
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
        locals()[str(county)+'_sum'][year-1980,0] = year
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(DD)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_BloomGDD4.csv', locals()[str(county)+'_sum'], delimiter = ',')



##GrowingGDD4.5

for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmax.'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year)+'.nc')
    for county in county_list:
        subsettmax = ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax.air_temperature[1]
        if roitmax.air_temperature.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        datatmax = roitmax.air_temperature[fall_start:fall_end+1].values-273.15
        subsettmin = ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        datatmin = roitmin.air_temperature[fall_start:fall_end+1].values-273.15
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
        locals()[str(county)+'_sum'][year-1980,0] = year
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(DD)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_GrowingGDD4.csv', locals()[str(county)+'_sum'], delimiter = ',')



## Bloom >12.8
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmax.'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year)+'.nc')
    for county in county_list:
        subsettmax = ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax.air_temperature[1]
        if roitmax.air_temperature.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.air_temperature[fall_start:fall_end+1].values-273.15
        subsettmin = ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)                    
        datatmin = roitmin.air_temperature[fall_start:fall_end+1].values-273.15
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
        locals()[str(county)+'_sum'][year-1980,0] = year           
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_T12.8.csv', locals()[str(county)+'_sum'], delimiter = ',')

## Bloom >15.6
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmax.'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year)+'.nc')
    for county in county_list:
        subsettmax = ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax.air_temperature[1]
        if roitmax.air_temperature.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.air_temperature[fall_start:fall_end+1].values-273.15
        subsettmin = ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)                    
        datatmin = roitmin.air_temperature[fall_start:fall_end+1].values-273.15
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
        locals()[str(county)+'_sum'][year-1980,0] = year           
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_T15.6.csv', locals()[str(county)+'_sum'], delimiter = ',')

## Bloom >=18.3 <= 26.7
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmax.'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year)+'.nc')
    for county in county_list:
        subsettmax = ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax.air_temperature[1]
        if roitmax.air_temperature.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.air_temperature[fall_start:fall_end+1].values-273.15
        subsettmin = ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)                    
        datatmin = roitmin.air_temperature[fall_start:fall_end+1].values-273.15
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
        locals()[str(county)+'_sum'][year-1980,0] = year           
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_T18.3-26.7.csv', locals()[str(county)+'_sum'], delimiter = ',')


## Bloom >=10 <= 21.1
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmax.'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year)+'.nc')
    for county in county_list:
        subsettmax = ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax.air_temperature[1]
        if roitmax.air_temperature.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.air_temperature[fall_start:fall_end+1].values-273.15
        subsettmin = ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)                    
        datatmin = roitmin.air_temperature[fall_start:fall_end+1].values-273.15
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
        locals()[str(county)+'_sum'][year-1980,0] = year           
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_T10_21.1.csv', locals()[str(county)+'_sum'], delimiter = ',')



## Bloom >=21.1 <= 30.6
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmax.'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year)+'.nc')
    for county in county_list:
        subsettmax = ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax.air_temperature[1]
        if roitmax.air_temperature.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.air_temperature[fall_start:fall_end+1].values-273.15
        subsettmin = ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)                    
        datatmin = roitmin.air_temperature[fall_start:fall_end+1].values-273.15
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
        locals()[str(county)+'_sum'][year-1980,0] = year           
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_T21.1-30.6.csv', locals()[str(county)+'_sum'], delimiter = ',')


## Bloom >4.4
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmax.'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year)+'.nc')
    for county in county_list:
        subsettmax = ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax.air_temperature[1]
        if roitmax.air_temperature.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.air_temperature[fall_start:fall_end+1].values-273.15
        subsettmin = ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)                    
        datatmin = roitmin.air_temperature[fall_start:fall_end+1].values-273.15
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
        locals()[str(county)+'_sum'][year-1980,0] = year           
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_T4.4.csv', locals()[str(county)+'_sum'], delimiter = ',')



## Bloom >32.2
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmax.'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year)+'.nc')
    for county in county_list:
        subsettmax = ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax.air_temperature[1]
        if roitmax.air_temperature.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.air_temperature[fall_start:fall_end+1].values-273.15
        subsettmin = ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)                    
        datatmin = roitmin.air_temperature[fall_start:fall_end+1].values-273.15
        sumdays = np.zeros((datatmax.shape[0], datatmax.shape[1], datatmax.shape[2]))
        for day in range(0, datatmax.shape[0]):
            for lat in range(0,datatmax.shape[1]):
                for lon in range(0,datatmax.shape[2]):
                    if (datatmax[day, lat, lon]+datatmin[day, lat, lon])/2 > 32.2:
                        sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis = 0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan        
        locals()[str(county)+'_sum'][year-1980,0] = year           
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_T32.2.csv', locals()[str(county)+'_sum'], delimiter = ',')
    
    

## Mar Tmin
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year)+'.nc')
    for county in county_list:
        subsettmin = ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)                    
        shapedata = roitmin.air_temperature[1]
        if roitmin.air_temperature.shape[0] == 366:
            fall_start = 60
            fall_end = 90
        else:
            fall_start = 59
            fall_end = 89
        datatmin = roitmin.air_temperature[fall_start:fall_end+1].values
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-1980,0] = year           
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(Tmindata)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_MarTmin.csv', locals()[str(county)+'_sum'], delimiter = ',')
    

## Dormancy Chill
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmax.'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year)+'.nc')
    pre_ncdatatmax = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmax.'+str(year-1)+'.nc')
    pre_ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year-1)+'.nc')
    for county in county_list:
        subsettmax = ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax.air_temperature[1]
        if roitmax.air_temperature.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        Tmaxdata = roitmax.air_temperature[fall_start:fall_end+1].values-273.15
        subsettmin = ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        Tmindata = roitmin.air_temperature[fall_start:fall_end+1].values-273.15
        subsettmax = pre_ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax.air_temperature[1]
        if roitmax.air_temperature.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        pre_Tmaxdata = roitmax.air_temperature[fall_start:fall_end+1].values-273.15
        subsettmin = pre_ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        pre_Tmindata = roitmin.air_temperature[fall_start:fall_end+1].values-273.15

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

        WC = np.nansum(WC, axis = 0)
        pre_WC = np.nansum(pre_WC, axis = 0)
        WC[np.isinf(WC)] = np.nan
        pre_WC[np.isinf(pre_WC)] = np.nan
        WC = WC + pre_WC
        for lat in range(0,WC.shape[0]):
            for lon in range(0, WC.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    WC[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1980,0] = year
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(WC)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_DormancyChill.csv', locals()[str(county)+'_sum'], delimiter = ',')

##DormancyFreeze
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmax.'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year)+'.nc')
    pre_ncdatatmax = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmax.'+str(year-1)+'.nc')
    pre_ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year-1)+'.nc')
    for county in county_list:
        subsettmax = ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax.air_temperature[1]
        if roitmax.air_temperature.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        Tmaxdata = roitmax.air_temperature[fall_start:fall_end+1].values-273.15
        subsettmin = ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        Tmindata = roitmin.air_temperature[fall_start:fall_end+1].values-273.15
        subsettmax = pre_ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax.air_temperature[1]
        if roitmax.air_temperature.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        pre_Tmaxdata = roitmax.air_temperature[fall_start:fall_end+1].values-273.15
        subsettmin = pre_ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        pre_Tmindata = roitmin.air_temperature[fall_start:fall_end+1].values-273.15

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

        WC = np.nansum(WC, axis = 0)
        pre_WC = np.nansum(pre_WC, axis = 0)
        WC[np.isinf(WC)] = np.nan
        pre_WC[np.isinf(pre_WC)] = np.nan
        WC = WC + pre_WC
        for lat in range(0,WC.shape[0]):
            for lon in range(0, WC.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    WC[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1981,0] = year
        locals()[str(county)+'_sum'][year-1981,1] = np.nanmean(WC)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_DormancyFreeze.csv', locals()[str(county)+'_sum'], delimiter = ',')



## Bloom sph
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatsph = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.SpH._'+str(year)+'.nc')
    for county in county_list:
        subset = ncdatatsph.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi.specific_humidity[1]
        if roi.specific_humidity.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        data = roi.specific_humidity[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-1980,0] = year   
        data = np.nanmean(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_SpH.csv', locals()[str(county)+'_sum'], delimiter = ',')


## Bloom Wnd
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatsph = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.WndSpd._'+str(year)+'.nc')
    for county in county_list:
        subset = ncdatatsph.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi.wind_speed[1]
        if roi.wind_speed.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        data = roi.wind_speed[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-1980,0] = year   
        data = np.nanmean(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(data)*3.6
for county in county_list:
    savetxt(str(save_path)+str(county)+'_WndSpd.csv', locals()[str(county)+'_sum'], delimiter = ',')



##Feb ppt
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2)) 
for year in range(1980, 2021):
    ncdata = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Precip.'+str(year)+'.nc')
    for county in county_list:
        subset = ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi.precipitation_amount[1]
        if roi.precipitation_amount.shape[0] == 366:
            fall_start = 31
            fall_end = 59
        else:
            fall_start = 31
            fall_end = 58
        data = roi.precipitation_amount[fall_start:fall_end+1].values
        locals()[str(county)+'_sum'][year-1980,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_FebPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')
    
    
## Feb Tmin
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year)+'.nc')
    for county in county_list:
        subsettmin = ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)                    
        shapedata = roitmin.air_temperature[1]
        if roitmin.air_temperature.shape[0] == 366:
            fall_start = 31
            fall_end = 59
        else:
            fall_start = 31
            fall_end = 58
        datatmin = roitmin.air_temperature[fall_start:fall_end+1].values
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan                    
        locals()[str(county)+'_sum'][year-1980,0] = year           
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(Tmindata)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_FebTmin.csv', locals()[str(county)+'_sum'], delimiter = ',')
    
    
    

##Bloom ETo
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    pre_ncdata = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.ETo._'+str(year-1)+'.nc')
    ncdata = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.ETo._'+str(year)+'.nc')
    for county in county_list:
        subset = pre_ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi.potential_evapotranspiration[1]
        if roi.potential_evapotranspiration.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        data = roi.potential_evapotranspiration[fall_start:fall_end+1].values
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1980,0] = year
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_BloomETo.csv', locals()[str(county)+'_sum'], delimiter = ',')


##Growing ETo
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    pre_ncdata = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.ETo._'+str(year-1)+'.nc')
    ncdata = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.ETo._'+str(year)+'.nc')
    for county in county_list:
        subset = pre_ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi.potential_evapotranspiration[1]
        if roi.potential_evapotranspiration.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        data = roi.potential_evapotranspiration[fall_start:fall_end+1].values
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1980,0] = year
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_GrowingETo.csv', locals()[str(county)+'_sum'], delimiter = ',')

for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmin = salem.open_xr_dataset('/group/moniergrp/GridMet/GridMet_mask_no_almond/CA_GridMet.Tmin.'+str(year)+'.nc')
    for county in county_list:
        subsettmin = ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmin.air_temperature[1]
        if roitmin.air_temperature.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmin = roitmin.air_temperature[fall_start:fall_end+1].values
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1980,0] = year
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(Tmindata)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_BloomTmin.csv', locals()[str(county)+'_sum'], delimiter = ',')

