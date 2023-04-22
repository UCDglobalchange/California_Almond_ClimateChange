## code to collect gridmet ACIs and save into a single netCDF file

import pandas as pd
import numpy as np
from numpy import genfromtxt
from numpy import savetxt
import numpy as np
import pandas as pd
import netCDF4 as nc
from numpy import genfromtxt
import math

data_ID='11_19'
load_path = '/home/shqwu/Almond_code_git/saved_data/'+str(data_ID)+'/Gridmet_ACI/'
save_path = '/home/shqwu/Almond_code_git/saved_data/'+str(data_ID)+'/Gridmet_nc/'
# create netcdf file
Almond = nc.Dataset(str(save_path)+'gridmet_ACI.nc', 'w', format = 'NETCDF4')
yield_csv = genfromtxt('/home/shqwu/Almond_code_git/almond_yield_1980_2020.csv', delimiter = ',')[:,1:]

## define dimensions
Almond.createDimension('Time', size = 41)
Almond.createDimension('Region', size =  16)
Almond.createDimension('ACI', size = 13)
## define variables
Years = Almond.createVariable('Year', 'f4', dimensions = 'Time')
Countys = Almond.createVariable('County', str, dimensions = 'Region')
ACIs = Almond.createVariable('ACI_name', 'U', dimensions = 'ACI')
ACI_list = ['DormancyFreeze','DormancyETo','JanPpt','BloomPpt','BloomTmin' ,'BloomETo', 'BloomGDD4','SpH','WndSpd','GrowingETo','GrowingGDD4', 'GrowingKDD30','harvest_Ppt']

Yields = Almond.createVariable('Yield', 'f4', dimensions = ('Time', 'Region'))
ACI_values = Almond.createVariable('ACI_value', 'f4', dimensions = ('Time', 'Region','ACI'))
county_list = ['Butte', 'Colusa', 'Fresno', 'Glenn', 'Kern', 'Kings', 'Madera', 'Merced', 'San Joaquin', 'Solano', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Yolo', 'Yuba']                      
ACI_name_unit  = ['DormancyFreeze', 'DormancyTmean','DormancyETo', 'JanPpt','FebTmin','BloomPpt', 'BloomETo', 'BloomGDD4','SpH','WndSpd','MarT-2', 'GrowingETo', 'GrowingTmean', 'GrowingKDD30','harvest_Ppt']


##add data
Years[:] = np.arange(1980,2021)
for i in range(0,16):
    Countys[i] = county_list[i]
for i in range(0,13):
    ACIs[i] = ACI_name_unit[i]

for county_id in range(0,16):
    for ACI_id in range(0,13):
        ACI_id_name = ACI_list[ACI_id]
        ACI_id_data = genfromtxt(str(load_path)+str(county_list[county_id])+'_'+str(ACI_id_name)+'.csv', delimiter = ',')
        ACI_id_data = ACI_id_data[:,1]
        for year_id in range(0,41):
            ACI_values[year_id, county_id, ACI_id] = ACI_id_data[year_id]
Almond.close()



