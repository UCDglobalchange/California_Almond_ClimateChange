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


home_path='/home/shqwu/California_Almond_ClimateChange-main/Run_project'
input_path = home_path+'/input_data/GridMet/'
save_path = home_path+'/intermediate_data/Gridmet_ACI/'
shp_path = home_path+'/input_data/CA_Counties/'


county_list = ['Butte', 'Colusa', 'Fresno', 'Glenn', 'Kern', 'Kings', 'Madera', 'Merced', 'San Joaquin', 'Solano', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Yolo', 'Yuba']                      
shapefile = salem.read_shapefile(shp_path+'Counties.shp')
for county in county_list:
     locals()[str(county)+'_shp'] = shapefile.loc[shapefile['NAME'].isin([str(county)])]

#harvest ETo
var_name = 'potential_evapotranspiration'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))    
for year in range(1980, 2021):
    ncdata = salem.open_xr_dataset(input_path+'pet_'+str(year)+'.nc')
    for county in county_list:
        subset = getattr(( ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi[1]
        if roi.shape[0] == 366:
            fall_start = 213 
            fall_end = 304
        else:
            fall_start = 212
            fall_end = 303
        data = roi[fall_start:fall_end+1].values
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
var_name = 'precipitation_amount'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdata = salem.open_xr_dataset(input_path+'pr_'+str(year)+'.nc')
    for county in county_list:
        subset = getattr(( ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi[1]
        if roi.shape[0] == 366:
            fall_start = 213
            fall_end = 304
        else:
            fall_start = 212
            fall_end = 303
        data = roi[fall_start:fall_end+1].values
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
var_name = 'precipitation_amount'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdata = salem.open_xr_dataset(input_path+'pr_'+str(year)+'.nc')
    for county in county_list:
        subset = getattr(( ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi[1]
        if roi.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        data = roi[fall_start:fall_end+1].values
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
var_name = 'air_temperature'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset(input_path+'tmmx_'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset(input_path+'tmmn_'+str(year)+'.nc')
    for county in county_list:
        subsettmax = getattr((ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax[1]
        if roitmax.shape[0] == 366:
            fall_start = 214
            fall_end = 304
        else:
            fall_start = 213
            fall_end = 303
        datatmax = roitmax[fall_start:fall_end+1].values
        Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmaxdata[lat,lon] = np.nan
        subsettmin = getattr((ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)                    
        datatmin = roitmin[fall_start:fall_end+1].values
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
var_name = 'precipitation_amount'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdata = salem.open_xr_dataset(input_path+'pr_'+str(year)+'.nc')
    for county in county_list:
        subset = getattr(( ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi[1]
        if roi.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        data = roi[fall_start:fall_end+1].values
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
var_name = 'air_temperature'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset(input_path+'tmmx_'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset(input_path+'tmmn_'+str(year)+'.nc')
    for county in county_list:
        subsettmax = getattr((ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax[1]
        if roitmax.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax[fall_start:fall_end+1].values
        Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmaxdata[lat,lon] = np.nan
        subsettmin = getattr((ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)                    
        datatmin = roitmin[fall_start:fall_end+1].values
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
var_name = 'precipitation_amount'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdata = salem.open_xr_dataset(input_path+'pr_'+str(year)+'.nc')
    for county in county_list:
        subset = getattr(( ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi[1]
        if roi.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        data = roi[fall_start:fall_end+1].values
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
var_name = 'air_temperature'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset(input_path+'tmmx_'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset(input_path+'tmmn_'+str(year)+'.nc')
    for county in county_list:
        subsettmax = getattr((ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax[1]
        if roitmax.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        datatmax = roitmax[fall_start:fall_end+1].values
        Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmaxdata[lat,lon] = np.nan
        subsettmin = getattr((ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)                    
        datatmin = roitmin[fall_start:fall_end+1].values
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
var_name = 'precipitation_amount'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    pre_ncdata = salem.open_xr_dataset(input_path+'pr_'+str(year-1)+'.nc')
    ncdata = salem.open_xr_dataset(input_path+'pr_'+str(year)+'.nc')
    for county in county_list:
        subset = getattr((ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi[1]
        if roi.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        data = roi[fall_start:fall_end+1].values
        pre_data = np.nansum(data, axis = 0)
        
        subset = getattr((pre_ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi[1]
        if roi.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        data = roi[fall_start:fall_end+1].values
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
var_name = 'potential_evapotranspiration'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    pre_ncdata = salem.open_xr_dataset(input_path+'pet_'+str(year-1)+'.nc')
    ncdata = salem.open_xr_dataset(input_path+'pet_'+str(year)+'.nc')
    for county in county_list:
        subset = getattr(( ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi[1]
        if roi.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        data = roi[fall_start:fall_end+1].values
        pre_data = np.nansum(data, axis = 0)        
        subset = getattr((pre_ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi[1]
        if roi.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        data = roi[fall_start:fall_end+1].values
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
var_name = 'air_temperature'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset(input_path+'tmmx_'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset(input_path+'tmmn_'+str(year)+'.nc')
    pre_ncdatatmax = salem.open_xr_dataset(input_path+'tmmx_'+str(year-1)+'.nc')
    pre_ncdatatmin = salem.open_xr_dataset(input_path+'tmmn_'+str(year-1)+'.nc')
    for county in county_list:
        subsettmax = getattr((ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax[1]
        if roitmax.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        datatmax = roitmax[fall_start:fall_end+1].values
        Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmaxdata[lat,lon] = np.nan
        subsettmin = getattr((ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)                    
        datatmin = roitmin[fall_start:fall_end+1].values
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan
        
        subsettmax = getattr((pre_ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax[1]
        if roitmax.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        datatmax = roitmax[fall_start:fall_end+1].values
        pre_Tmaxdata = np.nanmean(datatmax, axis = 0)-273.15
        for lat in range(0,Tmaxdata.shape[0]):
            for lon in range(0, Tmaxdata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    pre_Tmaxdata[lat,lon] = np.nan
        subsettmin = getattr((pre_ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)                    
        datatmin = roitmin[fall_start:fall_end+1].values
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
var_name = 'air_temperature'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset(input_path+'tmmx_'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset(input_path+'tmmn_'+str(year)+'.nc')
    for county in county_list:
        subsettmax = getattr((ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax[1]
        if roitmax.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        datatmax = roitmax[fall_start:fall_end+1].values-273.15
        subsettmin = getattr((ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        datatmin = roitmin[fall_start:fall_end+1].values-273.15
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
var_name = 'air_temperature'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset(input_path+'tmmx_'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset(input_path+'tmmn_'+str(year)+'.nc')
    for county in county_list:
        subsettmax = getattr((ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax[1]
        if roitmax.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax[fall_start:fall_end+1].values-273.15
        subsettmin = getattr((ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        datatmin = roitmin[fall_start:fall_end+1].values-273.15
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
var_name = 'air_temperature'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset(input_path+'tmmx_'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset(input_path+'tmmn_'+str(year)+'.nc')
    for county in county_list:
        subsettmax = getattr((ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax[1]
        if roitmax.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax[fall_start:fall_end+1].values-273.15
        subsettmin = getattr((ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        datatmin = roitmin[fall_start:fall_end+1].values-273.15
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
var_name = 'air_temperature'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset(input_path+'tmmx_'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset(input_path+'tmmn_'+str(year)+'.nc')
    for county in county_list:
        subsettmax = getattr((ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax[1]
        if roitmax.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        datatmax = roitmax[fall_start:fall_end+1].values-273.15
        subsettmin = getattr((ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        datatmin = roitmin[fall_start:fall_end+1].values-273.15
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





## Dormancy Chill
var_name = 'air_temperature'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset(input_path+'tmmx_'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset(input_path+'tmmn_'+str(year)+'.nc')
    pre_ncdatatmax = salem.open_xr_dataset(input_path+'tmmx_'+str(year-1)+'.nc')
    pre_ncdatatmin = salem.open_xr_dataset(input_path+'tmmn_'+str(year-1)+'.nc')
    for county in county_list:
        subsettmax = getattr((ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax[1]
        if roitmax.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        Tmaxdata = roitmax[fall_start:fall_end+1].values-273.15
        subsettmin = getattr((ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        Tmindata = roitmin[fall_start:fall_end+1].values-273.15
        subsettmax = getattr((pre_ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax[1]
        if roitmax.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        pre_Tmaxdata = roitmax[fall_start:fall_end+1].values-273.15
        subsettmin = getattr((pre_ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        pre_Tmindata = roitmin[fall_start:fall_end+1].values-273.15

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
var_name = 'air_temperature'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmax = salem.open_xr_dataset(input_path+'tmmx_'+str(year)+'.nc')
    ncdatatmin = salem.open_xr_dataset(input_path+'tmmn_'+str(year)+'.nc')
    pre_ncdatatmax = salem.open_xr_dataset(input_path+'tmmx_'+str(year-1)+'.nc')
    pre_ncdatatmin = salem.open_xr_dataset(input_path+'tmmn_'+str(year-1)+'.nc')
    for county in county_list:
        subsettmax = getattr((ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax[1]
        if roitmax.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        Tmaxdata = roitmax[fall_start:fall_end+1].values-273.15
        subsettmin = getattr((ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        Tmindata = roitmin[fall_start:fall_end+1].values-273.15
        subsettmax = getattr((pre_ncdatatmax.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmax = subsettmax.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmax[1]
        if roitmax.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        pre_Tmaxdata = roitmax[fall_start:fall_end+1].values-273.15
        subsettmin = getattr((pre_ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        pre_Tmindata = roitmin[fall_start:fall_end+1].values-273.15

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
var_name = 'specific_humidity'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatsph = salem.open_xr_dataset(input_path+'sph_'+str(year)+'.nc')
    for county in county_list:
        subset = getattr((ncdatatsph.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi[1]
        if roi.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        data = roi[fall_start:fall_end+1]
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
var_name = 'wind_speed'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatsph = salem.open_xr_dataset(input_path+'vs_'+str(year)+'.nc')
    for county in county_list:
        subset = getattr((ncdatatsph.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi[1]
        if roi.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        data = roi[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-1980,0] = year   
        data = np.nanmean(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(data)*3.6
for county in county_list:
    savetxt(str(save_path)+str(county)+'_WndSpd.csv', locals()[str(county)+'_sum'], delimiter = ',')




    

##Bloom ETo
var_name = 'potential_evapotranspiration'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    pre_ncdata = salem.open_xr_dataset(input_path+'pet_'+str(year-1)+'.nc')
    ncdata = salem.open_xr_dataset(input_path+'pet_'+str(year)+'.nc')
    for county in county_list:
        subset = getattr((pre_ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)),var_name)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi[1]
        if roi.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        data = roi[fall_start:fall_end+1].values
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
var_name = 'potential_evapotranspiration'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    pre_ncdata = salem.open_xr_dataset(input_path+'pet_'+str(year-1)+'.nc')
    ncdata = salem.open_xr_dataset(input_path+'pet_'+str(year)+'.nc')
    for county in county_list:
        subset = getattr((pre_ncdata.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roi = subset.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roi[1]
        if roi.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        data = roi[fall_start:fall_end+1].values
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1980,0] = year
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_GrowingETo.csv', locals()[str(county)+'_sum'], delimiter = ',')

var_name = 'air_temperature'
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((41,2))
for year in range(1980, 2021):
    ncdatatmin = salem.open_xr_dataset(input_path+'tmmn_'+str(year)+'.nc')
    for county in county_list:
        subsettmin = getattr((ncdatatmin.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0)), var_name)
        roitmin = subsettmin.salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        shapedata = roitmin[1]
        if roitmax.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmin = roitmin[fall_start:fall_end+1].values
        Tmindata = np.nanmean(datatmin, axis = 0)-273.15
        for lat in range(0,Tmindata.shape[0]):
            for lon in range(0, Tmindata.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    Tmindata[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1980,0] = year
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(Tmindata)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_BloomTmin.csv', locals()[str(county)+'_sum'], delimiter = ',')

