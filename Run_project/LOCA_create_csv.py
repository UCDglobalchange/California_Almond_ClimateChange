## Code to normalize LOCA ACIs 
## Code to save normalized ACI and their squares to csv 

import math 
import pandas as pd
import numpy as np
import netCDF4 as nc
from math import sqrt
from sklearn import preprocessing
from numpy import genfromtxt
from numpy import savetxt


county_list = ['Butte', 'Colusa', 'Fresno', 'Glenn', 'Kern', 'Kings', 'Madera', 'Merced', 'San Joaquin', 'Solano', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Yolo', 'Yuba']                      

aci_num = 14
CM_num = 8
home_path='~/Run_project'
input_path = home_path+'/intermediate_data/LOCA_nc/'
save_path = home_path+'/intermediate_data/LOCA_csv/'

model_list =  ['ACCESS-CM2', 'EC-Earth3', 'EC-Earth3-Veg', 'GFDL-ESM4',  'INM-CM5-0',  'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'CNRM-ESM2-1']

middle_tech_scenario = np.zeros(80)
middle_tech_scenario[0] = 41
for i in range(1,80):
    middle_tech_scenario[i] = middle_tech_scenario[i-1] + (80 - i)/80
for model_id in range(0,CM_num):
    print(model_id)
    almond_hist = nc.Dataset(input_path+str(model_list[model_id])+'_hist_ACI.nc')
    almond_ssp245 = nc.Dataset(input_path+str(model_list[model_id])+'_ssp245_ACI.nc')
    ACI_ssp245_1980_2020_sum = np.zeros((0,aci_num*2+32))
    ACI_ssp245_s_1980_2020_sum = np.zeros((0,aci_num*2+32))
    ACI_ssp245_m_1980_2020_sum = np.zeros((0,aci_num*2+32))
    ACI_ssp245_2021_2099_sum = np.zeros((0,aci_num*2+32))
    ACI_ssp245_s_2021_2099_sum = np.zeros((0,aci_num*2+32))
    ACI_ssp245_m_2021_2099_sum = np.zeros((0,aci_num*2+32))
    aci_sum_state_hist = np.zeros((16*35, aci_num))
    aci_sum_state_ssp245 = np.zeros((16*6, aci_num))
    for i in range(aci_num):
        aci_sum_state_hist[:,i] = np.ndarray.flatten(almond_hist.variables['ACI_value'][:,:,i])
        aci_sum_state_ssp245[:,i] = np.ndarray.flatten(almond_ssp245.variables['ACI_value'][0:6,:,i])
    aci_sum_state_hist_ssp245 = np.row_stack((aci_sum_state_hist, aci_sum_state_ssp245))
    aci_sum_state_hist_ssp245_sq = aci_sum_state_hist_ssp245**2
    aci_mean_ssp245 = np.mean(aci_sum_state_hist_ssp245, axis = 0)
    aci_std_ssp245 = np.std(aci_sum_state_hist_ssp245, axis = 0)
    aci_mean_ssp245_sq = np.mean(aci_sum_state_hist_ssp245_sq, axis=0)
    aci_std_ssp245_sq = np.std(aci_sum_state_hist_ssp245_sq, axis=0)
    for region_id in range(0,16):
        print(region_id)
        non_clim_coef_hist = np.zeros((35,32))
        non_clim_coef_hist[:,region_id] = np.arange(1,36,1)
        non_clim_coef_hist[:,(region_id+16)] = 1
        aci_hist = np.array(almond_hist.variables['ACI_value'][:,region_id,:], dtype = np.float64)
        aci_hist = np.column_stack((aci_hist, aci_hist**2))
        aci_ssp245 = np.array(almond_ssp245.variables['ACI_value'][:,region_id,:], dtype = np.float64)
        aci_ssp245 = np.column_stack((aci_ssp245, aci_ssp245**2))
        non_clim_coef= np.zeros((85,32))
        non_clim_coef[:,region_id] = np.arange(36,121,1)
        non_clim_coef[:,(region_id+16)] = 1
        non_clim_coef_s = np.zeros((85, 32))
        non_clim_coef_s[0:6,region_id] = np.arange(36,42,1)
        non_clim_coef_s[6:,region_id] = 41
        non_clim_coef_s[:,(region_id+16)] = 1
        non_clim_coef_m = np.zeros((85, 32))
        non_clim_coef_m[0:6,region_id] = np.arange(36,42,1)
        non_clim_coef_m[6:,region_id] = middle_tech_scenario[1:]
        non_clim_coef_m[:,(region_id+16)] = 1
        ACI_ssp245_1980_2020 = np.column_stack((np.row_stack((aci_hist, aci_ssp245[0:6])), np.row_stack((non_clim_coef_hist, non_clim_coef[0:6]))))
        ACI_ssp245_s_1980_2020 = np.column_stack((np.row_stack((aci_hist, aci_ssp245[0:6])), np.row_stack((non_clim_coef_hist, non_clim_coef_s[0:6]))))
        ACI_ssp245_m_1980_2020 = np.column_stack((np.row_stack((aci_hist, aci_ssp245[0:6])), np.row_stack((non_clim_coef_hist, non_clim_coef_m[0:6]))))
        ACI_ssp245_2021_2099 = np.column_stack((aci_ssp245[6:85], non_clim_coef[6:85]))
        ACI_ssp245_s_2021_2099 = np.column_stack((aci_ssp245[6:85], non_clim_coef_s[6:85]))
        ACI_ssp245_m_2021_2099 = np.column_stack((aci_ssp245[6:85], non_clim_coef_m[6:85]))
        for j in range(0,aci_num):
            ACI_ssp245_2021_2099[:,j] = (ACI_ssp245_2021_2099[:,j]-aci_mean_ssp245[j])/aci_std_ssp245[j]
            ACI_ssp245_s_2021_2099[:,j] = (ACI_ssp245_s_2021_2099[:,j]-aci_mean_ssp245[j])/aci_std_ssp245[j]
            ACI_ssp245_m_2021_2099[:,j] = (ACI_ssp245_m_2021_2099[:,j]-aci_mean_ssp245[j])/aci_std_ssp245[j]
            ACI_ssp245_1980_2020[:,j] = (ACI_ssp245_1980_2020[:,j]-aci_mean_ssp245[j])/aci_std_ssp245[j]
            ACI_ssp245_s_1980_2020[:,j] = (ACI_ssp245_s_1980_2020[:,j]-aci_mean_ssp245[j])/aci_std_ssp245[j]
            ACI_ssp245_m_1980_2020[:,j] = (ACI_ssp245_m_1980_2020[:,j]-aci_mean_ssp245[j])/aci_std_ssp245[j]
            ACI_ssp245_2021_2099[:,j+aci_num] = (ACI_ssp245_2021_2099[:,j+aci_num]-aci_mean_ssp245_sq[j])/aci_std_ssp245_sq[j]
            ACI_ssp245_s_2021_2099[:,j+aci_num] = (ACI_ssp245_s_2021_2099[:,j+aci_num]-aci_mean_ssp245_sq[j])/aci_std_ssp245_sq[j]
            ACI_ssp245_m_2021_2099[:,j+aci_num] = (ACI_ssp245_m_2021_2099[:,j+aci_num]-aci_mean_ssp245_sq[j])/aci_std_ssp245_sq[j]
            ACI_ssp245_1980_2020[:,j+aci_num] = (ACI_ssp245_1980_2020[:,j+aci_num]-aci_mean_ssp245_sq[j])/aci_std_ssp245_sq[j]
            ACI_ssp245_s_1980_2020[:,j+aci_num] = (ACI_ssp245_s_1980_2020[:,j+aci_num]-aci_mean_ssp245_sq[j])/aci_std_ssp245_sq[j]
            ACI_ssp245_m_1980_2020[:,j+aci_num] = (ACI_ssp245_m_1980_2020[:,j+aci_num]-aci_mean_ssp245_sq[j])/aci_std_ssp245_sq[j]
        ACI_ssp245_1980_2020_sum = np.row_stack((ACI_ssp245_1980_2020_sum, ACI_ssp245_1980_2020))
        ACI_ssp245_s_1980_2020_sum = np.row_stack((ACI_ssp245_s_1980_2020_sum,ACI_ssp245_s_1980_2020))
        ACI_ssp245_m_1980_2020_sum = np.row_stack((ACI_ssp245_m_1980_2020_sum,ACI_ssp245_m_1980_2020))
        ACI_ssp245_2021_2099_sum = np.row_stack((ACI_ssp245_2021_2099_sum,ACI_ssp245_2021_2099))
        ACI_ssp245_s_2021_2099_sum = np.row_stack((ACI_ssp245_s_2021_2099_sum,ACI_ssp245_s_2021_2099))
        ACI_ssp245_m_2021_2099_sum = np.row_stack((ACI_ssp245_m_2021_2099_sum,ACI_ssp245_m_2021_2099))
        print(non_clim_coef_m)
        print(middle_tech_scenario)
    savetxt(str(save_path)+str(model_list[model_id])+'hist_ssp245_ACI.csv', ACI_ssp245_1980_2020_sum, delimiter = ',')
    savetxt(str(save_path)+str(model_list[model_id])+'hist_ssp245_s_ACI.csv', ACI_ssp245_s_1980_2020_sum, delimiter = ',')
    savetxt(str(save_path)+str(model_list[model_id])+'hist_ssp245_m_ACI.csv', ACI_ssp245_m_1980_2020_sum, delimiter = ',')
    savetxt(str(save_path)+str(model_list[model_id])+'future_ssp245_ACI.csv', ACI_ssp245_2021_2099_sum, delimiter = ',')
    savetxt(str(save_path)+str(model_list[model_id])+'future_ssp245_s_ACI.csv', ACI_ssp245_s_2021_2099_sum, delimiter = ',')
    savetxt(str(save_path)+str(model_list[model_id])+'future_ssp245_m_ACI.csv', ACI_ssp245_m_2021_2099_sum, delimiter = ',')

middle_tech_scenario = np.zeros(80)
middle_tech_scenario[0] = 41
for i in range(1,80):
    middle_tech_scenario[i] = middle_tech_scenario[i-1] + (80 - i)/80
for model_id in range(0,CM_num):
    print(model_id)
    almond_hist = nc.Dataset(input_path+str(model_list[model_id])+'_hist_ACI.nc')
    almond_ssp585 = nc.Dataset(input_path+str(model_list[model_id])+'_ssp585_ACI.nc')
    ACI_ssp585_1980_2020_sum = np.zeros((0,aci_num*2+32))
    ACI_ssp585_s_1980_2020_sum = np.zeros((0,aci_num*2+32))
    ACI_ssp585_m_1980_2020_sum = np.zeros((0,aci_num*2+32))
    ACI_ssp585_2021_2099_sum = np.zeros((0,aci_num*2+32))
    ACI_ssp585_s_2021_2099_sum = np.zeros((0,aci_num*2+32))
    ACI_ssp585_m_2021_2099_sum = np.zeros((0,aci_num*2+32))
    aci_sum_state_hist = np.zeros((16*35, aci_num))
    aci_sum_state_ssp585 = np.zeros((16*6, aci_num))
    for i in range(aci_num):
        aci_sum_state_hist[:,i] = np.ndarray.flatten(almond_hist.variables['ACI_value'][:,:,i])
        aci_sum_state_ssp585[:,i] = np.ndarray.flatten(almond_ssp585.variables['ACI_value'][0:6,:,i])
    aci_sum_state_hist_ssp585 = np.row_stack((aci_sum_state_hist, aci_sum_state_ssp585))
    aci_sum_state_hist_ssp585_sq = aci_sum_state_hist_ssp585**2
    aci_mean_ssp585 = np.mean(aci_sum_state_hist_ssp585, axis = 0)
    aci_std_ssp585 = np.std(aci_sum_state_hist_ssp585, axis = 0)
    aci_mean_ssp585_sq = np.mean(aci_sum_state_hist_ssp585_sq, axis=0)
    aci_std_ssp585_sq = np.std(aci_sum_state_hist_ssp585_sq, axis=0)
    for region_id in range(0,16):
        print(region_id)
        non_clim_coef_hist = np.zeros((35,32))
        non_clim_coef_hist[:,region_id] = np.arange(1,36,1)
        non_clim_coef_hist[:,(region_id+16)] = 1
        aci_hist = np.array(almond_hist.variables['ACI_value'][:,region_id,:], dtype = np.float64)
        aci_hist = np.column_stack((aci_hist, aci_hist**2))
        aci_ssp585 = np.array(almond_ssp585.variables['ACI_value'][:,region_id,:], dtype = np.float64)
        aci_ssp585 = np.column_stack((aci_ssp585, aci_ssp585**2))
        non_clim_coef= np.zeros((85,32))
        non_clim_coef[:,region_id] = np.arange(36,121,1)
        non_clim_coef[:,(region_id+16)] = 1
        non_clim_coef_s = np.zeros((85, 32))
        non_clim_coef_s[0:6,region_id] = np.arange(36,42,1)
        non_clim_coef_s[6:,region_id] = 41
        non_clim_coef_s[:,(region_id+16)] = 1
        non_clim_coef_m = np.zeros((85, 32))
        non_clim_coef_m[0:6,region_id] = np.arange(36,42,1)
        non_clim_coef_m[6:,region_id] = middle_tech_scenario[1:]
        non_clim_coef_m[:,(region_id+16)] = 1
        ACI_ssp585_1980_2020 = np.column_stack((np.row_stack((aci_hist, aci_ssp585[0:6])), np.row_stack((non_clim_coef_hist, non_clim_coef[0:6]))))
        ACI_ssp585_s_1980_2020 = np.column_stack((np.row_stack((aci_hist, aci_ssp585[0:6])), np.row_stack((non_clim_coef_hist, non_clim_coef_s[0:6]))))
        ACI_ssp585_m_1980_2020 = np.column_stack((np.row_stack((aci_hist, aci_ssp585[0:6])), np.row_stack((non_clim_coef_hist, non_clim_coef_m[0:6]))))
        ACI_ssp585_2021_2099 = np.column_stack((aci_ssp585[6:85], non_clim_coef[6:85]))
        ACI_ssp585_s_2021_2099 = np.column_stack((aci_ssp585[6:85], non_clim_coef_s[6:85]))
        ACI_ssp585_m_2021_2099 = np.column_stack((aci_ssp585[6:85], non_clim_coef_m[6:85]))
        for j in range(0,aci_num):
            ACI_ssp585_2021_2099[:,j] = (ACI_ssp585_2021_2099[:,j]-aci_mean_ssp585[j])/aci_std_ssp585[j]
            ACI_ssp585_s_2021_2099[:,j] = (ACI_ssp585_s_2021_2099[:,j]-aci_mean_ssp585[j])/aci_std_ssp585[j]
            ACI_ssp585_m_2021_2099[:,j] = (ACI_ssp585_m_2021_2099[:,j]-aci_mean_ssp585[j])/aci_std_ssp585[j]
            ACI_ssp585_1980_2020[:,j] = (ACI_ssp585_1980_2020[:,j]-aci_mean_ssp585[j])/aci_std_ssp585[j]
            ACI_ssp585_s_1980_2020[:,j] = (ACI_ssp585_s_1980_2020[:,j]-aci_mean_ssp585[j])/aci_std_ssp585[j]
            ACI_ssp585_m_1980_2020[:,j] = (ACI_ssp585_m_1980_2020[:,j]-aci_mean_ssp585[j])/aci_std_ssp585[j]
            ACI_ssp585_2021_2099[:,j+aci_num] = (ACI_ssp585_2021_2099[:,j+aci_num]-aci_mean_ssp585_sq[j])/aci_std_ssp585_sq[j]
            ACI_ssp585_s_2021_2099[:,j+aci_num] = (ACI_ssp585_s_2021_2099[:,j+aci_num]-aci_mean_ssp585_sq[j])/aci_std_ssp585_sq[j]
            ACI_ssp585_m_2021_2099[:,j+aci_num] = (ACI_ssp585_m_2021_2099[:,j+aci_num]-aci_mean_ssp585_sq[j])/aci_std_ssp585_sq[j]
            ACI_ssp585_1980_2020[:,j+aci_num] = (ACI_ssp585_1980_2020[:,j+aci_num]-aci_mean_ssp585_sq[j])/aci_std_ssp585_sq[j]
            ACI_ssp585_s_1980_2020[:,j+aci_num] = (ACI_ssp585_s_1980_2020[:,j+aci_num]-aci_mean_ssp585_sq[j])/aci_std_ssp585_sq[j]
            ACI_ssp585_m_1980_2020[:,j+aci_num] = (ACI_ssp585_m_1980_2020[:,j+aci_num]-aci_mean_ssp585_sq[j])/aci_std_ssp585_sq[j]
        ACI_ssp585_1980_2020_sum = np.row_stack((ACI_ssp585_1980_2020_sum, ACI_ssp585_1980_2020))
        ACI_ssp585_s_1980_2020_sum = np.row_stack((ACI_ssp585_s_1980_2020_sum,ACI_ssp585_s_1980_2020))
        ACI_ssp585_m_1980_2020_sum = np.row_stack((ACI_ssp585_m_1980_2020_sum,ACI_ssp585_m_1980_2020))
        ACI_ssp585_2021_2099_sum = np.row_stack((ACI_ssp585_2021_2099_sum,ACI_ssp585_2021_2099))
        ACI_ssp585_s_2021_2099_sum = np.row_stack((ACI_ssp585_s_2021_2099_sum,ACI_ssp585_s_2021_2099))
        ACI_ssp585_m_2021_2099_sum = np.row_stack((ACI_ssp585_m_2021_2099_sum,ACI_ssp585_m_2021_2099))
        print(non_clim_coef_m)
        print(middle_tech_scenario)
    savetxt(str(save_path)+str(model_list[model_id])+'hist_ssp585_ACI.csv', ACI_ssp585_1980_2020_sum, delimiter = ',')
    savetxt(str(save_path)+str(model_list[model_id])+'hist_ssp585_s_ACI.csv', ACI_ssp585_s_1980_2020_sum, delimiter = ',')
    savetxt(str(save_path)+str(model_list[model_id])+'hist_ssp585_m_ACI.csv', ACI_ssp585_m_1980_2020_sum, delimiter = ',')
    savetxt(str(save_path)+str(model_list[model_id])+'future_ssp585_ACI.csv', ACI_ssp585_2021_2099_sum, delimiter = ',')
    savetxt(str(save_path)+str(model_list[model_id])+'future_ssp585_s_ACI.csv', ACI_ssp585_s_2021_2099_sum, delimiter = ',')
    savetxt(str(save_path)+str(model_list[model_id])+'future_ssp585_m_ACI.csv', ACI_ssp585_m_2021_2099_sum, delimiter = ',')

