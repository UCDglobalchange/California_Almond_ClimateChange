##Code to normalize gridmet data and save them into csv
## standardize aci by state mean and std
## sqaure ACI then std ACI2
import math
import pandas as pd
import numpy as np
import netCDF4 as nc
from math import sqrt
from sklearn import preprocessing
from numpy import genfromtxt
from numpy import savetxt
county_list = ['Butte', 'Colusa', 'Fresno', 'Glenn', 'Kern', 'Kings', 'Madera', 'Merced', 'San Joaquin', 'Solano', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Yolo', 'Yuba']                      

home_path='~/Run_project'
input_path = home_path+'/intermediate_data/Gridmet_nc/'
save_path = home_path+'/intermediate_data/Gridmet_csv/'

almond_hist = nc.Dataset(input_path+'gridmet_ACI.nc')
aci_num=14

aci_sum_state = np.zeros((16*41,aci_num))

for i in range(aci_num):
    aci_sum_state[:,i] = np.ndarray.flatten(almond_hist.variables['ACI_value'][:,:,i])
aci_sum_state_sq = aci_sum_state**2
aci_state_mean = np.mean(aci_sum_state, axis=0)
aci_state_std = np.std(aci_sum_state, axis=0)
aci_state_mean_sq = np.mean(aci_sum_state_sq, axis=0)
aci_state_std_sq = np.std(aci_sum_state_sq, axis=0)


ACI_sum = np.zeros((0,aci_num*2))
non_clim_coef_sum = np.zeros((0,32))
for region_id in range(0,16):
    non_clim_coef = np.zeros((41,32))
    non_clim_coef[:,region_id] = np.arange(1,42,1)
    non_clim_coef[:,(region_id+16)] = 1
    aci = np.array(almond_hist.variables['ACI_value'][:,region_id,:], dtype = float)
    aci_sq = aci**2
    for j in range(0,aci_num):
      aci[:,j] = (aci[:,j] - aci_state_mean[j]) / aci_state_std[j]
      aci_sq[:,j] = (aci_sq[:,j] - aci_state_mean_sq[j]) / aci_state_std_sq[j]
    aci = np.column_stack((aci, aci_sq))
    ACI_sum = np.row_stack((ACI_sum, aci))
    non_clim_coef_sum = np.row_stack((non_clim_coef_sum, non_clim_coef))
X = np.column_stack((ACI_sum,non_clim_coef_sum))
savetxt(save_path+'Gridmet.csv', X, delimiter = ',')

