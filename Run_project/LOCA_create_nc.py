## Code to collect all  LOCA ACIs and save them into netCDF file for each LOCA model

import pandas as pd
import numpy as np
from numpy import genfromtxt
from numpy import savetxt
import numpy as np
import pandas as pd
import netCDF4 as nc
from numpy import genfromtxt

model='CNRM-ESM2-1'
home_path='~/Run_project'
save_path = home_path+'/intermediate_data/LOCA_nc/'
input_path_ACI = home_path+'/intermediate_data/LOCA_ACI/'+str(model)+'/'
input_path_yield = home_path+'/input_data/'



aci_num=14
# create netcdf file
Almond = nc.Dataset(save_path+str(model)+'_hist_ACI.nc', 'w', format = 'NETCDF4')
yield_csv = genfromtxt(input_path_yield+'/almond_yield_1980_2020.csv', delimiter = ',')

## define dimensions
Almond.createDimension('Time', size = 35)
Almond.createDimension('Region', size =  16)
Almond.createDimension('ACI', size = aci_num)
## define variables
Years = Almond.createVariable('Year', 'f4', dimensions = 'Time')
Countys = Almond.createVariable('County', str, dimensions = 'Region')
ACIs = Almond.createVariable('ACI_name', 'U', dimensions = 'ACI')


ACI_list = ['DormancyChill','DormancyETo','JanPpt','BloomPpt','BloomTmin','BloomFrostDays' ,'BloomETo', 'BloomGDD','SpH','windspeed','GrowingETo','GrowingGDD', 'GrowingKDD','harvest_Ppt']

Yields = Almond.createVariable('Yield', 'f4', dimensions = ('Time', 'Region'))
ACI_values = Almond.createVariable('ACI_value', 'f4', dimensions = ('Time', 'Region','ACI'))
county_list = ['Butte', 'Colusa', 'Fresno', 'Glenn', 'Kern', 'Kings', 'Madera', 'Merced', 'San Joaquin', 'Solano', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Yolo', 'Yuba']                      
ACI_name_unit=  ['DormancyFreeze', 'DormancyTmean','DormancyETo', 'JanPpt','FebTmin','BloomPpt','BloomFrostDays', 'BloomETo', 'BloomGDD4','SpH','windspeed','MarTmin', 'GrowingETo', 'GrowingGDD4', 'GrowingKDD30','harvest_Ppt']


##add data
Years[:] = np.arange(1980,2015)
for i in range(0,16):
    Countys[i] = county_list[i]
for i in range(0,aci_num):
    ACIs[i] = ACI_name_unit[i]

for county_id in range(0,16):
    for ACI_id in range(0,aci_num):
        ACI_id_name = ACI_list[ACI_id]
        ACI_id_data = genfromtxt(input_path_ACI+str(county_list[county_id])+'_historical_'+str(ACI_id_name)+'.csv', delimiter = ',')[:,1]
        ACI_values[:, county_id, ACI_id] = ACI_id_data[:]
Almond.close()


Almond_ssp245 = nc.Dataset(str(save_path)+str(model)+'_ssp245_ACI.nc', 'w', format = 'NETCDF4')
## define dimensions
Almond_ssp245.createDimension('Time', size = 85)
Almond_ssp245.createDimension('Region', size =  16)
Almond_ssp245.createDimension('ACI', size = aci_num)
Years = Almond_ssp245.createVariable('Year', 'f4', dimensions = 'Time')
Countys = Almond_ssp245.createVariable('County', str, dimensions = 'Region')
ACIs = Almond_ssp245.createVariable('ACI_name', 'U', dimensions = 'ACI')
ACI_values = Almond_ssp245.createVariable('ACI_value', 'f4', dimensions = ('Time', 'Region','ACI'))

Years[:] = np.arange(2015,2100)
for i in range(0,16):
    Countys[i] = county_list[i]
for i in range(0,aci_num):
    ACIs[i] = ACI_name_unit[i]

for county_id in range(0,16):
    for ACI_id in range(0,aci_num):
        ACI_id_name = ACI_list[ACI_id]
        ACI_id_data = genfromtxt(input_path_ACI+str(county_list[county_id])+'_ssp245_'+str(ACI_id_name)+'.csv', delimiter = ',')[:,1]
        ACI_values[:, county_id, ACI_id] = ACI_id_data[:]
Almond_ssp245.close()

Almond_ssp585 = nc.Dataset(str(save_path)+str(model)+'_ssp585_ACI.nc', 'w', format = 'NETCDF4')
## define dimensions
Almond_ssp245.createDimension('Time', size = 85)
Almond_ssp245.createDimension('Region', size =  16)
Almond_ssp245.createDimension('ACI', size = aci_num)
Years = Almond_ssp245.createVariable('Year', 'f4', dimensions = 'Time')
Countys = Almond_ssp245.createVariable('County', str, dimensions = 'Region')
ACIs = Almond_ssp245.createVariable('ACI_name', 'U', dimensions = 'ACI')
ACI_values = Almond_ssp245.createVariable('ACI_value', 'f4', dimensions = ('Time', 'Region','ACI'))

Years[:] = np.arange(2015,2100)
for i in range(0,16):
    Countys[i] = county_list[i]
for i in range(0,aci_num):
    ACIs[i] = ACI_name_unit[i]

for county_id in range(0,16):
    for ACI_id in range(0,aci_num):
        ACI_id_name = ACI_list[ACI_id]
        ACI_id_data = genfromtxt(input_path_ACI+str(county_list[county_id])+'_ssp585_'+str(ACI_id_name)+'.csv', delimiter = ',')[:,1]
        ACI_values[:, county_id, ACI_id] = ACI_id_data[:]
Almond_ssp585.close()
