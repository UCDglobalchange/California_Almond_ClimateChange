## Code to collect all  MACA ACIs and save them into netCDF file for each MACA model

import pandas as pd
import numpy as np
from numpy import genfromtxt
from numpy import savetxt
import numpy as np
import pandas as pd
import netCDF4 as nc
from numpy import genfromtxt

model='MIROC-ESM-CHEM'

home_path=
save_path = home_path+'/intermediate_data/MACA_nc/'
input_path_ACI = home_path+'/intermediate_data/MACA_ACI/'+str(model)+'/'
input_path_yield = home_path+'/input_data/'



aci_num=13
# create netcdf file
Almond = nc.Dataset(save_path+str(model)+'_hist_ACI.nc', 'w', format = 'NETCDF4')
yield_csv = genfromtxt(input_path_yield+'/almond_yield_1980_2020.csv', delimiter = ',')

## define dimensions
Almond.createDimension('Time', size = 55)
Almond.createDimension('Region', size =  16)
Almond.createDimension('ACI', size = aci_num)
## define variables
Years = Almond.createVariable('Year', 'f4', dimensions = 'Time')
Countys = Almond.createVariable('County', str, dimensions = 'Region')
ACIs = Almond.createVariable('ACI_name', 'U', dimensions = 'ACI')


ACI_list = ['DormancyFreeze','DormancyETo','JanPpt','BloomPpt','BloomTmin' ,'BloomETo','SpH', 'BloomGDD4','windspeed','GrowingETo','GrowingGDD4', 'GrowingKDD30','harvest_Ppt']

Yields = Almond.createVariable('Yield', 'f4', dimensions = ('Time', 'Region'))
ACI_values = Almond.createVariable('ACI_value', 'f4', dimensions = ('Time', 'Region','ACI'))
county_list = ['Butte', 'Colusa', 'Fresno', 'Glenn', 'Kern', 'Kings', 'Madera', 'Merced', 'San Joaquin', 'Solano', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Yolo', 'Yuba']                      
ACI_name_unit=  ['DormancyFreeze', 'DormancyTmean','DormancyETo', 'JanPpt','FebTmin','BloomPpt', 'BloomETo', 'BloomGDD4','SpH','windspeed','MarTmin', 'GrowingETo', 'GrowingGDD4', 'GrowingKDD30','harvest_Ppt']


##add data
Years[:] = np.arange(1951,2006)
for i in range(0,16):
    Countys[i] = county_list[i]
for i in range(0,aci_num):
    ACIs[i] = ACI_name_unit[i]

for county_id in range(0,16):
    for ACI_id in range(0,aci_num):
        ACI_id_name = ACI_list[ACI_id]
        ACI_id_data = genfromtxt(input_path_ACI+str(county_list[county_id])+'_hist_'+str(ACI_id_name)+'.csv', delimiter = ',')
        ACI_len = len(ACI_id_data)-56
        ACI_id_data = ACI_id_data[ACI_len+1:][:,1]
        for year_id in range(0,55):
            ACI_values[year_id, county_id, ACI_id] = ACI_id_data[year_id]
Almond.close()


Almond_rcp45 = nc.Dataset(str(save_path)+str(model)+'_rcp45_ACI.nc', 'w', format = 'NETCDF4')
## define dimensions
Almond_rcp45.createDimension('Time', size = 94)
Almond_rcp45.createDimension('Region', size =  16)
Almond_rcp45.createDimension('ACI', size = aci_num)
Years = Almond_rcp45.createVariable('Year', 'f4', dimensions = 'Time')
Countys = Almond_rcp45.createVariable('County', str, dimensions = 'Region')
ACIs = Almond_rcp45.createVariable('ACI_name', 'U', dimensions = 'ACI')
ACI_values = Almond_rcp45.createVariable('ACI_value', 'f4', dimensions = ('Time', 'Region','ACI'))

Years[:] = np.arange(2006,2100)
for i in range(0,16):
    Countys[i] = county_list[i]
for i in range(0,aci_num):
    ACIs[i] = ACI_name_unit[i]

for county_id in range(0,16):
    for ACI_id in range(0,aci_num):
        ACI_id_name = ACI_list[ACI_id]
        ACI_id_data = genfromtxt(input_path_ACI+str(county_list[county_id])+'_rcp45_'+str(ACI_id_name)+'.csv', delimiter = ',')[:,1]
        for year_id in range(0,94):
            ACI_values[year_id, county_id, ACI_id] = ACI_id_data[year_id]
Almond_rcp45.close()

Almond_rcp85 = nc.Dataset(str(save_path)+str(model)+'_rcp85_ACI.nc', 'w', format = 'NETCDF4')
## define dimensions
Almond_rcp85.createDimension('Time', size = 94)
Almond_rcp85.createDimension('Region', size =  16)
Almond_rcp85.createDimension('ACI', size = aci_num)
Years = Almond_rcp85.createVariable('Year', 'f4', dimensions = 'Time')
Countys = Almond_rcp85.createVariable('County', str, dimensions = 'Region')
ACIs = Almond_rcp85.createVariable('ACI_name', 'U', dimensions = 'ACI')
ACI_values = Almond_rcp85.createVariable('ACI_value', 'f4', dimensions = ('Time', 'Region','ACI'))

Years[:] = np.arange(2006,2100)
for i in range(0,16):
    Countys[i] = county_list[i]
for i in range(0,aci_num):
    ACIs[i] = ACI_name_unit[i]

for county_id in range(0,16):
    for ACI_id in range(0,aci_num):
        print(ACI_id)
        ACI_id_name = ACI_list[ACI_id]
        ACI_id_data = genfromtxt(input_path_ACI+str(county_list[county_id])+'_rcp85_'+str(ACI_id_name)+'.csv', delimiter = ',')[:,1]
        for year_id in range(0,94):
            ACI_values[year_id, county_id, ACI_id] = ACI_id_data[year_id]
Almond_rcp85.close()



