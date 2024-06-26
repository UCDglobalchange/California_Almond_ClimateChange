## Code to calculate ACIs for each LOCA model

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
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    args = vars(parser.parse_args())
    return args

args = get_args()

model_name=str(args['model'])
print(model_name)
home_path='~/Run_project'
save_path = home_path+'/intermediate_data/LOCA_ACI/'+str(model_name)+'/'
shp_path = home_path+'/input_data/CA_Counties/'
input_path = home_path+'/input_data/LOCA/'
reference_cropland_path = home_path+'/input_data/reference_cropland/'

county_list = ['Butte', 'Colusa', 'Fresno', 'Glenn', 'Kern', 'Kings', 'Madera', 'Merced', 'San Joaquin', 'Solano', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Yolo', 'Yuba']                      
shapefile = salem.read_shapefile(shp_path+'Counties.shp')
for county in county_list:
     locals()[str(county)+'_shp'] = shapefile.loc[shapefile['NAME'].isin([str(county)])]
cropland_reference = salem.open_xr_dataset(reference_cropland_path+'LOCA_reference_cropland.nc')


var_list = ['pr', 'tasmin', 'tasmax','huss', 'hursmax', 'hursmin', 'rsds', 'wspeed', 'uas', 'vas']

for county in county_list:
    locals()[str(county)+'_reference'] = cropland_reference.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0).salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False).tasmax

for var in var_list:
    xr_data = salem.open_xr_dataset(input_path+str(model_name)+'_historical_r1i1p1f1_'+str(var)+'.nc')
    for county in county_list:
        xr_data_county_roi = xr_data.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0).salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        for year in range(1979,2015):
            day_start = np.where(pd.to_datetime(xr_data_county_roi.time.values).year==year)[0][0]
            day_end = np.where(pd.to_datetime(xr_data_county_roi.time.values).year==year)[0][-1]
            locals()[str(county)+str(var)+str(year)+'_roi'] = xr_data_county_roi[var][day_start:(day_end+1)]

for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((35,2))
for year in range(1980, 2015):
    for county in county_list:
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        shapedata = locals()[str(county)+'_reference'].values
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
        locals()[str(county)+'_sum'][year-1980,0] = year
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(Tmindata)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_historical_BloomTmin.csv', locals()[str(county)+'_sum'], delimiter = ',')



for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((35,2))
for year in range(1980, 2015):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        shapedata =  locals()[str(county)+'_reference'].values
        if roitmax.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        roitmax = locals()[str(county)+'tasmax'+str(year-1)+'_roi']
        if roitmax.values.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        pre_Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year-1)+'_roi']
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
        locals()[str(county)+'_sum'][year-1980,0] = year
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(WC)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_historical_DormancyChill.csv', locals()[str(county)+'_sum'], delimiter = ',')



##dormancy ETO
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((35,2))
for year in range(1980, 2015):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        lat = roitmax.lat.values
        shapedata =  locals()[str(county)+'_reference'].values
        if roitmax.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15

        huss = locals()[str(county)+'huss'+str(year)+'_roi'][fall_start:fall_end+1].values #kg/kg
        hursmax = locals()[str(county)+'hursmax'+str(year)+'_roi'][fall_start:fall_end+1].values #%
        hursmin = locals()[str(county)+'hursmin'+str(year)+'_roi'][fall_start:fall_end+1].values
        rsds =( locals()[str(county)+'rsds'+str(year)+'_roi'][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
        uas = locals()[str(county)+'uas'+str(year)+'_roi'][fall_start:fall_end+1].values # m/s at 10m
        vas = locals()[str(county)+'vas'+str(year)+'_roi'][fall_start:fall_end+1].values

        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        u_2m = 0.748*((uas**2+vas**2)**0.5)
        Tmean = (Tmaxdata+Tmindata)/2
        D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
        j = np.pi*lat/180
        g = 0.066
        e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
        e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
        es = (e0_min+e0_max)/2
        ea = (e0_min*hursmax/100+e0_max*hursmin/100)/2
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
        pre_ETo = (0.408*D*Rn+g*900*u_2m*(es-ea)/(Tmean+273.16))/(D+g*(1+0.34*u_2m))
        roitmax = locals()[str(county)+'tasmax'+str(year-1)+'_roi']
        if roitmax.values.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15

        huss = locals()[str(county)+'huss'+str(year-1)+'_roi'][fall_start:fall_end+1].values #kg/kg
        hursmax = locals()[str(county)+'hursmax'+str(year-1)+'_roi'][fall_start:fall_end+1].values #%
        hursmin = locals()[str(county)+'hursmin'+str(year-1)+'_roi'][fall_start:fall_end+1].values
        rsds =( locals()[str(county)+'rsds'+str(year-1)+'_roi'][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
        uas = locals()[str(county)+'uas'+str(year-1)+'_roi'][fall_start:fall_end+1].values # m/s at 10m
        vas = locals()[str(county)+'vas'+str(year-1)+'_roi'][fall_start:fall_end+1].values


        roitmin = locals()[str(county)+'tasmin'+str(year-1)+'_roi']
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        u_2m = 0.748*((uas**2+vas**2)**0.5)
        Tmean = (Tmaxdata+Tmindata)/2
        D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
        j = np.pi*lat/180
        g = 0.066
        e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
        e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
        es = (e0_min+e0_max)/2
        ea = (e0_min*hursmax/100+e0_max*hursmin/100)/2
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
        ETo = (0.408*D*Rn+g*900*u_2m*(es-ea)/(Tmean+273.16))/(D+g*(1+0.34*u_2m))

        ETo_sum = np.row_stack((pre_ETo,ETo))
        ETo_day_num = ETo_sum.shape[0]
        ETo_sum = np.nanmean(ETo_sum,axis = 0)
        for lat in range(0,ETo_sum.shape[0]):
            for lon in range(0, ETo_sum.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    ETo_sum[lat,lon] = np.nan

        locals()[str(county)+'_sum'][year-1980,0] = year
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(ETo_sum)*ETo_day_num
for county in county_list:
    savetxt(str(save_path)+str(county)+'_historical_DormancyETo.csv', locals()[str(county)+'_sum'], delimiter = ',')








    


#bloom sph
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((35,2))
for year in range(1980, 2015):
    for county in county_list:
        roitmin = locals()[str(county)+'huss'+str(year)+'_roi']
        shapedata =  locals()[str(county)+'_reference'].values
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
        locals()[str(county)+'_sum'][year-1980,0] = year           
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(Tmindata)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_historical_SpH.csv', locals()[str(county)+'_sum'], delimiter = ',')

for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((35,2)) 
for year in range(1980, 2015):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        shapedata = locals()[str(county)+'_reference'].values
        if roi.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        data = roi.values[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-1980,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_historical_JanPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')



##harvest ppt
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((35,2))
for year in range(1980, 2015):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        shapedata =  locals()[str(county)+'_reference'].values
        if roi.values.shape[0] == 366:
            fall_start = 244
            fall_end = 334
        else:
            fall_start = 243
            fall_end = 333
        data = roi.values[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-1980,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_historical_harvest_Ppt.csv', locals()[str(county)+'_sum'], delimiter = ',')









## Growing KDD30
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((35,2))
for year in range(1980, 2015):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        shapedata = locals()[str(county)+'_reference'].values
        if roitmax.values.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        DD = np.zeros((datatmin.shape[0],datatmin.shape[1], datatmin.shape[2]))
        for day in range(0,datatmin.shape[0]):
            for lat in range(0,datatmin.shape[1]):
                for lon in range(0, datatmin.shape[2]):
                    if np.isnan(datatmax[day, lat,lon]):
                        pass
                    else:
                        b = 35
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
    savetxt(str(save_path)+str(county)+'_historical_GrowingKDD.csv', locals()[str(county)+'_sum'], delimiter = ',')





#bloom  ETo
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((35,2))
for year in range(1980, 2015):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        lat = roitmax.lat.values
        shapedata = locals()[str(county)+'_reference'].values
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        huss = locals()[str(county)+'huss'+str(year)+'_roi'][fall_start:fall_end+1].values #kg/kg
        hursmax = locals()[str(county)+'hursmax'+str(year)+'_roi'][fall_start:fall_end+1].values #%
        hursmin = locals()[str(county)+'hursmin'+str(year)+'_roi'][fall_start:fall_end+1].values
        rsds =( locals()[str(county)+'rsds'+str(year)+'_roi'][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
        uas = locals()[str(county)+'uas'+str(year)+'_roi'][fall_start:fall_end+1].values # m/s at 10m
        vas = locals()[str(county)+'vas'+str(year)+'_roi'][fall_start:fall_end+1].values

        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15
        
        u_2m = 0.748*((uas**2+vas**2)**0.5)
        Tmean = (Tmaxdata+Tmindata)/2
        D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
        j = np.pi*lat/180
        g = 0.066
        e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
        e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
        es = (e0_min+e0_max)/2
        ea = (e0_min*hursmax/100+e0_max*hursmin/100)/2
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
        ETo = (0.408*D*Rn+g*900*u_2m*(es-ea)/(Tmean+273.16))/(D+g*(1+0.34*u_2m))        
        ETo_day_num = ETo.shape[0]
        ETo = np.nanmean(ETo,axis = 0)
        for lat in range(0,ETo.shape[0]):
            for lon in range(0, ETo.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    ETo[lat,lon] = np.nan
                locals()[str(county)+'_sum'][year-1980,0] = year           
                locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(ETo)*ETo_day_num
for county in county_list:
    savetxt(str(save_path)+str(county)+'_historical_BloomETo.csv', locals()[str(county)+'_sum'], delimiter = ',')


#growing eto
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((35,2))
for year in range(1980, 2015):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        lat = roitmax.lat.values
        shapedata = locals()[str(county)+'_reference'].values
        if roitmax.values.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        huss = locals()[str(county)+'huss'+str(year)+'_roi'][fall_start:fall_end+1].values #kg/kg
        hursmax = locals()[str(county)+'hursmax'+str(year)+'_roi'][fall_start:fall_end+1].values #%
        hursmin = locals()[str(county)+'hursmin'+str(year)+'_roi'][fall_start:fall_end+1].values
        rsds =( locals()[str(county)+'rsds'+str(year)+'_roi'][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
        uas = locals()[str(county)+'uas'+str(year)+'_roi'][fall_start:fall_end+1].values # m/s at 10m
        vas = locals()[str(county)+'vas'+str(year)+'_roi'][fall_start:fall_end+1].values

        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15
        
        u_2m = 0.748*((uas**2+vas**2)**0.5)
        Tmean = (Tmaxdata+Tmindata)/2
        D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
        j = np.pi*lat/180
        g = 0.066
        e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
        e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
        es = (e0_min+e0_max)/2
        ea = (e0_min*hursmax/100+e0_max*hursmin/100)/2
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
        ETo = (0.408*D*Rn+g*900*u_2m*(es-ea)/(Tmean+273.16))/(D+g*(1+0.34*u_2m))        
        ETo_day_num = ETo.shape[0]
        ETo = np.nanmean(ETo,axis = 0)
        for lat in range(0,ETo.shape[0]):
            for lon in range(0, ETo.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    ETo[lat,lon] = np.nan
                locals()[str(county)+'_sum'][year-1980,0] = year           
                locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(ETo)*ETo_day_num
for county in county_list:
    savetxt(str(save_path)+str(county)+'_historical_GrowingETo.csv', locals()[str(county)+'_sum'], delimiter = ',')



## bloom gdd4.5
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((35,2))
for year in range(1980, 2015):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        shapedata = locals()[str(county)+'_reference'].values
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
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
        locals()[str(county)+'_sum'][year-1980,0] = year
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(DD)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_historical_BloomGDD.csv', locals()[str(county)+'_sum'], delimiter = ',')




##grpwomg gdd4.5
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((35,2))
for year in range(1980, 2015):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        shapedata = locals()[str(county)+'_reference'].values
        if roitmax.values.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        DD = np.zeros((datatmin.shape[0],datatmin.shape[1], datatmin.shape[2]))
        for day in range(0,datatmin.shape[0]):
            for lat in range(0,datatmin.shape[1]):
                for lon in range(0, datatmin.shape[2]):
                    if np.isnan(datatmax[day, lat,lon]):
                        pass
                    else:
                        b = 4.5
                        u = 35
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
    savetxt(str(save_path)+str(county)+'_historical_GrowingGDD.csv', locals()[str(county)+'_sum'], delimiter = ',')


##bloom windspeed
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((35,2))
for year in range(1980, 2015):
    for county in county_list:
        roivas = locals()[str(county)+'vas'+str(year)+'_roi']
        roiuas = locals()[str(county)+'uas'+str(year)+'_roi']
        shapedata = locals()[str(county)+'_reference'].values
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
        locals()[str(county)+'_sum'][year-1980,0] = year
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_historical_windspeed.csv', locals()[str(county)+'_sum'], delimiter = ',')


for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((35,2))
for year in range(1980, 2015):
    for county in county_list:
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        shapedata = locals()[str(county)+'_reference'].values
        if roitmin.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        data = roitmin.values[fall_start:fall_end+1]-273.15
        sumdays = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
        for day in range(data.shape[0]):
            for lat in range(data.shape[1]):
                for lon in range(data.shape[2]):
                    if data[day,lat,lon] < 0:
                       sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis=0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1980,0] = year
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_historical_BloomFrostDays.csv', locals()[str(county)+'_sum'], delimiter = ',')


for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((35,2))
for year in range(1980, 2015):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        shapedata = locals()[str(county)+'_reference'].values
        if roi.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        data = roi.values[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-1980,0] = year
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-1980,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_historical_BloomPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')


for var in var_list:
    xr_data = salem.open_xr_dataset(input_path+str(model_name)+'_ssp245_r1i1p1f1_'+str(var)+'.nc')
    for county in county_list:
        xr_data_county_roi = xr_data.salem.subset(shape = locals()[str(county)+'_shp'], margin = 0).salem.roi(shape = locals()[str(county)+'_shp'], all_touched=False)
        for year in range(2015,2100):
            day_start = np.where(pd.to_datetime(xr_data_county_roi.time.values).year==year)[0][0]
            day_end = np.where(pd.to_datetime(xr_data_county_roi.time.values).year==year)[0][-1]
            locals()[str(county)+str(var)+str(year)+'_roi'] = xr_data_county_roi[var][day_start:(day_end+1)]


for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((85,2))
for year in range(2015, 2100):
    for county in county_list:
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        shapedata = locals()[str(county)+'_reference'].values
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
        locals()[str(county)+'_sum'][year-2015,0] = year
        locals()[str(county)+'_sum'][year-2015,1] = np.nanmean(Tmindata)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_ssp245_BloomTmin.csv', locals()[str(county)+'_sum'], delimiter = ',')



for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((85,2))
for year in range(2015, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        shapedata =  locals()[str(county)+'_reference'].values
        if roitmax.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        roitmax = locals()[str(county)+'tasmax'+str(year-1)+'_roi']
        if roitmax.values.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        pre_Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year-1)+'_roi']
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
        locals()[str(county)+'_sum'][year-2015,0] = year
        locals()[str(county)+'_sum'][year-2015,1] = np.nanmean(WC)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_ssp245_DormancyChill.csv', locals()[str(county)+'_sum'], delimiter = ',')



##dormancy ETO
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((85,2))
for year in range(2015, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        lat = roitmax.lat.values
        shapedata =  locals()[str(county)+'_reference'].values
        if roitmax.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15

        huss = locals()[str(county)+'huss'+str(year)+'_roi'][fall_start:fall_end+1].values #kg/kg
        hursmax = locals()[str(county)+'hursmax'+str(year)+'_roi'][fall_start:fall_end+1].values #%
        hursmin = locals()[str(county)+'hursmin'+str(year)+'_roi'][fall_start:fall_end+1].values
        rsds =( locals()[str(county)+'rsds'+str(year)+'_roi'][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
        uas = locals()[str(county)+'uas'+str(year)+'_roi'][fall_start:fall_end+1].values # m/s at 10m
        vas = locals()[str(county)+'vas'+str(year)+'_roi'][fall_start:fall_end+1].values

        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        u_2m = 0.748*((uas**2+vas**2)**0.5)
        Tmean = (Tmaxdata+Tmindata)/2
        D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
        j = np.pi*lat/180
        g = 0.066
        e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
        e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
        es = (e0_min+e0_max)/2
        ea = (e0_min*hursmax/100+e0_max*hursmin/100)/2
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
        pre_ETo = (0.408*D*Rn+g*900*u_2m*(es-ea)/(Tmean+273.16))/(D+g*(1+0.34*u_2m))
        roitmax = locals()[str(county)+'tasmax'+str(year-1)+'_roi']
        if roitmax.values.shape[0] == 366:
            fall_start = 305
            fall_end = 365
        else:
            fall_start = 304
            fall_end = 364
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15

        huss = locals()[str(county)+'huss'+str(year-1)+'_roi'][fall_start:fall_end+1].values #kg/kg
        hursmax = locals()[str(county)+'hursmax'+str(year-1)+'_roi'][fall_start:fall_end+1].values #%
        hursmin = locals()[str(county)+'hursmin'+str(year-1)+'_roi'][fall_start:fall_end+1].values
        rsds =( locals()[str(county)+'rsds'+str(year-1)+'_roi'][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
        uas = locals()[str(county)+'uas'+str(year-1)+'_roi'][fall_start:fall_end+1].values # m/s at 10m
        vas = locals()[str(county)+'vas'+str(year-1)+'_roi'][fall_start:fall_end+1].values


        roitmin = locals()[str(county)+'tasmin'+str(year-1)+'_roi']
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15

        u_2m = 0.748*((uas**2+vas**2)**0.5)
        Tmean = (Tmaxdata+Tmindata)/2
        D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
        j = np.pi*lat/180
        g = 0.066
        e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
        e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
        es = (e0_min+e0_max)/2
        ea = (e0_min*hursmax/100+e0_max*hursmin/100)/2
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
        ETo = (0.408*D*Rn+g*900*u_2m*(es-ea)/(Tmean+273.16))/(D+g*(1+0.34*u_2m))

        ETo_sum = np.row_stack((pre_ETo,ETo))
        ETo_day_num = ETo_sum.shape[0]
        ETo_sum = np.nanmean(ETo_sum,axis = 0)
        for lat in range(0,ETo_sum.shape[0]):
            for lon in range(0, ETo_sum.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    ETo_sum[lat,lon] = np.nan

        locals()[str(county)+'_sum'][year-2015,0] = year
        locals()[str(county)+'_sum'][year-2015,1] = np.nanmean(ETo_sum)*ETo_day_num
for county in county_list:
    savetxt(str(save_path)+str(county)+'_ssp245_DormancyETo.csv', locals()[str(county)+'_sum'], delimiter = ',')








    


#bloom sph
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((85,2))
for year in range(2015, 2100):
    for county in county_list:
        roitmin = locals()[str(county)+'huss'+str(year)+'_roi']
        shapedata =  locals()[str(county)+'_reference'].values
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
        locals()[str(county)+'_sum'][year-2015,0] = year           
        locals()[str(county)+'_sum'][year-2015,1] = np.nanmean(Tmindata)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_ssp245_SpH.csv', locals()[str(county)+'_sum'], delimiter = ',')

for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((85,2)) 
for year in range(2015, 2100):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        shapedata = locals()[str(county)+'_reference'].values
        if roi.values.shape[0] == 366:
            fall_start = 0
            fall_end = 30
        else:
            fall_start = 0
            fall_end = 30
        data = roi.values[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-2015,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2015,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_ssp245_JanPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')



##harvest ppt
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((85,2))
for year in range(2015, 2100):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        shapedata =  locals()[str(county)+'_reference'].values
        if roi.values.shape[0] == 366:
            fall_start = 244
            fall_end = 334
        else:
            fall_start = 243
            fall_end = 333
        data = roi.values[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-2015,0] = year   
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2015,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_ssp245_harvest_Ppt.csv', locals()[str(county)+'_sum'], delimiter = ',')









## Growing KDD30
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((85,2))
for year in range(2015, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        shapedata = locals()[str(county)+'_reference'].values
        if roitmax.values.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        DD = np.zeros((datatmin.shape[0],datatmin.shape[1], datatmin.shape[2]))
        for day in range(0,datatmin.shape[0]):
            for lat in range(0,datatmin.shape[1]):
                for lon in range(0, datatmin.shape[2]):
                    if np.isnan(datatmax[day, lat,lon]):
                        pass
                    else:
                        b = 35
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
        locals()[str(county)+'_sum'][year-2015,0] = year
        locals()[str(county)+'_sum'][year-2015,1] = np.nanmean(DD)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_ssp245_GrowingKDD.csv', locals()[str(county)+'_sum'], delimiter = ',')





#bloom  ETo
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((85,2))
for year in range(2015, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        lat = roitmax.lat.values
        shapedata = locals()[str(county)+'_reference'].values
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        huss = locals()[str(county)+'huss'+str(year)+'_roi'][fall_start:fall_end+1].values #kg/kg
        hursmax = locals()[str(county)+'hursmax'+str(year)+'_roi'][fall_start:fall_end+1].values #%
        hursmin = locals()[str(county)+'hursmin'+str(year)+'_roi'][fall_start:fall_end+1].values
        rsds =( locals()[str(county)+'rsds'+str(year)+'_roi'][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
        uas = locals()[str(county)+'uas'+str(year)+'_roi'][fall_start:fall_end+1].values # m/s at 10m
        vas = locals()[str(county)+'vas'+str(year)+'_roi'][fall_start:fall_end+1].values

        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15
        
        u_2m = 0.748*((uas**2+vas**2)**0.5)
        Tmean = (Tmaxdata+Tmindata)/2
        D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
        j = np.pi*lat/180
        g = 0.066
        e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
        e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
        es = (e0_min+e0_max)/2
        ea = (e0_min*hursmax/100+e0_max*hursmin/100)/2
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
        ETo = (0.408*D*Rn+g*900*u_2m*(es-ea)/(Tmean+273.16))/(D+g*(1+0.34*u_2m))        
        ETo_day_num = ETo.shape[0]
        ETo = np.nanmean(ETo,axis = 0)
        for lat in range(0,ETo.shape[0]):
            for lon in range(0, ETo.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    ETo[lat,lon] = np.nan
                locals()[str(county)+'_sum'][year-2015,0] = year           
                locals()[str(county)+'_sum'][year-2015,1] = np.nanmean(ETo)*ETo_day_num
for county in county_list:
    savetxt(str(save_path)+str(county)+'_ssp245_BloomETo.csv', locals()[str(county)+'_sum'], delimiter = ',')


#growing eto
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((85,2))
for year in range(2015, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        lat = roitmax.lat.values
        shapedata = locals()[str(county)+'_reference'].values
        if roitmax.values.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        Tmaxdata = roitmax.values[fall_start:fall_end+1]-273.15
        huss = locals()[str(county)+'huss'+str(year)+'_roi'][fall_start:fall_end+1].values #kg/kg
        hursmax = locals()[str(county)+'hursmax'+str(year)+'_roi'][fall_start:fall_end+1].values #%
        hursmin = locals()[str(county)+'hursmin'+str(year)+'_roi'][fall_start:fall_end+1].values
        rsds =( locals()[str(county)+'rsds'+str(year)+'_roi'][fall_start:fall_end+1].values)/(1e6/60/60/24) # W/m2
        uas = locals()[str(county)+'uas'+str(year)+'_roi'][fall_start:fall_end+1].values # m/s at 10m
        vas = locals()[str(county)+'vas'+str(year)+'_roi'][fall_start:fall_end+1].values

        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        Tmindata = roitmin.values[fall_start:fall_end+1]-273.15
        
        u_2m = 0.748*((uas**2+vas**2)**0.5)
        Tmean = (Tmaxdata+Tmindata)/2
        D = (4098*(0.6108*np.exp(17.27*Tmean/(Tmean+237.3))))/((Tmean+237.3)**2)
        j = np.pi*lat/180
        g = 0.066
        e0_min = 0.6108*np.exp(17.27*Tmindata/(Tmindata+237.3))
        e0_max = 0.6108*np.exp(17.27*Tmaxdata/(Tmaxdata+237.3))
        es = (e0_min+e0_max)/2
        ea = (e0_min*hursmax/100+e0_max*hursmin/100)/2
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
        ETo = (0.408*D*Rn+g*900*u_2m*(es-ea)/(Tmean+273.16))/(D+g*(1+0.34*u_2m))        
        ETo_day_num = ETo.shape[0]
        ETo = np.nanmean(ETo,axis = 0)
        for lat in range(0,ETo.shape[0]):
            for lon in range(0, ETo.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    ETo[lat,lon] = np.nan
                locals()[str(county)+'_sum'][year-2015,0] = year           
                locals()[str(county)+'_sum'][year-2015,1] = np.nanmean(ETo)*ETo_day_num
for county in county_list:
    savetxt(str(save_path)+str(county)+'_ssp245_GrowingETo.csv', locals()[str(county)+'_sum'], delimiter = ',')



## bloom gdd4.5
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((85,2))
for year in range(2015, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        shapedata = locals()[str(county)+'_reference'].values
        if roitmax.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
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
        locals()[str(county)+'_sum'][year-2015,0] = year
        locals()[str(county)+'_sum'][year-2015,1] = np.nanmean(DD)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_ssp245_BloomGDD.csv', locals()[str(county)+'_sum'], delimiter = ',')




##grpwomg gdd4.5
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((85,2))
for year in range(2015, 2100):
    for county in county_list:
        roitmax = locals()[str(county)+'tasmax'+str(year)+'_roi']
        shapedata = locals()[str(county)+'_reference'].values
        if roitmax.values.shape[0] == 366:
            fall_start = 60
            fall_end = 212
        else:
            fall_start = 59
            fall_end = 211
        datatmax = roitmax.values[fall_start:fall_end+1]-273.15
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        datatmin = roitmin.values[fall_start:fall_end+1]-273.15
        DD = np.zeros((datatmin.shape[0],datatmin.shape[1], datatmin.shape[2]))
        for day in range(0,datatmin.shape[0]):
            for lat in range(0,datatmin.shape[1]):
                for lon in range(0, datatmin.shape[2]):
                    if np.isnan(datatmax[day, lat,lon]):
                        pass
                    else:
                        b = 4.5
                        u = 35
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
        locals()[str(county)+'_sum'][year-2015,0] = year
        locals()[str(county)+'_sum'][year-2015,1] = np.nanmean(DD)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_ssp245_GrowingGDD.csv', locals()[str(county)+'_sum'], delimiter = ',')


##bloom windspeed
for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((85,2))
for year in range(2015, 2100):
    for county in county_list:
        roivas = locals()[str(county)+'vas'+str(year)+'_roi']
        roiuas = locals()[str(county)+'uas'+str(year)+'_roi']
        shapedata = locals()[str(county)+'_reference'].values
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
        locals()[str(county)+'_sum'][year-2015,0] = year
        locals()[str(county)+'_sum'][year-2015,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_ssp245_windspeed.csv', locals()[str(county)+'_sum'], delimiter = ',')


for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((85,2))
for year in range(2015, 2100):
    for county in county_list:
        roitmin = locals()[str(county)+'tasmin'+str(year)+'_roi']
        shapedata = locals()[str(county)+'_reference'].values
        if roitmin.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        data = roitmin.values[fall_start:fall_end+1]-273.15
        sumdays = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
        for day in range(data.shape[0]):
            for lat in range(data.shape[1]):
                for lon in range(data.shape[2]):
                    if data[day,lat,lon] < 0:
                       sumdays[day,lat,lon] = 1
        sumdays = np.nansum(sumdays, axis=0)
        for lat in range(0,sumdays.shape[0]):
            for lon in range(0, sumdays.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    sumdays[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2015,0] = year
        locals()[str(county)+'_sum'][year-2015,1] = np.nanmean(sumdays)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_ssp245_BloomFrostDays.csv', locals()[str(county)+'_sum'], delimiter = ',')


for county in county_list:
    locals()[str(county)+'_sum'] = np.zeros((85,2))
for year in range(2015, 2100):
    for county in county_list:
        roi = locals()[str(county)+'pr'+str(year)+'_roi']
        shapedata = locals()[str(county)+'_reference'].values
        if roi.values.shape[0] == 366:
            fall_start = 31
            fall_end = 74
        else:
            fall_start = 31
            fall_end = 73
        data = roi.values[fall_start:fall_end+1]
        locals()[str(county)+'_sum'][year-2015,0] = year
        data = np.nansum(data, axis = 0)
        for lat in range(0,data.shape[0]):
            for lon in range(0, data.shape[1]):
                if np.isnan(shapedata[lat,lon]):
                    data[lat,lon] = np.nan
        locals()[str(county)+'_sum'][year-2015,1] = np.nanmean(data)
for county in county_list:
    savetxt(str(save_path)+str(county)+'_ssp245_BloomPpt.csv', locals()[str(county)+'_sum'], delimiter = ',')






