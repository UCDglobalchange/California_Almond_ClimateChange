##Code to normalize gridmet data and save them into csv

import math
import pandas as pd
import numpy as np
import netCDF4 as nc
from math import sqrt
from sklearn import preprocessing
from numpy import genfromtxt
from numpy import savetxt
county_list = ['Butte', 'Colusa', 'Fresno', 'Glenn', 'Kern', 'Kings', 'Madera', 'Merced', 'San Joaquin', 'Solano', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Yolo', 'Yuba']                      

input_path = '../intermediate_data/Gridmet_nc/'
save_path = '../intermediate_data/Gridmet_csv/'

almond_hist = nc.Dataset(input_path+'gridmet_ACI.nc')
aci_num=13
ACI_sum = np.zeros((0,aci_num*2))
non_clim_coef_sum = np.zeros((0,32))
for region_id in range(0,16):
    non_clim_coef = np.zeros((41,32))
    non_clim_coef[:,region_id] = np.arange(1,42,1)
    non_clim_coef[:,(region_id+16)] = 1
    aci = np.array(almond_hist.variables['ACI_value'][:,region_id,:], dtype = np.float)
    aci_mean = np.mean(aci, axis = 0)
    aci_std = np.std(aci,axis=0)
    for j in range(0,aci_num):
      aci[:,j] = (aci[:,j] - aci_mean[j])/aci_std[j]
    aci = np.column_stack((aci, aci**2))
    ACI_sum = np.row_stack((ACI_sum, aci))
    non_clim_coef_sum = np.row_stack((non_clim_coef_sum, non_clim_coef))
X = np.column_stack((ACI_sum,non_clim_coef_sum))
savetxt(save_path+'Gridmet.csv', X, delimiter = ',')

