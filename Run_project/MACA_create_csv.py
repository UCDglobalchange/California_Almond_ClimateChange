## Code to normalize MACA ACIs 
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

home_path='~/Run_project'
input_path = home_path+'/intermediate_data/MACA_nc/'
save_path = home_path+'/intermediate_data/MACA_csv/'

model_list = ['bcc-csm1-1','bcc-csm1-1-m', 'BNU-ESM', 'CanESM2', 'CSIRO-Mk3-6-0', 'GFDL-ESM2G', 'GFDL-ESM2M', 'inmcm4', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR','CNRM-CM5', 'HadGEM2-CC365','HadGEM2-ES365', 'IPSL-CM5B-LR', 'MIROC5', 'MIROC-ESM', 'MIROC-ESM-CHEM', 'MRI-CGCM3']
middle_tech_scenario = np.zeros(80)
middle_tech_scenario[0] = 41
for i in range(1,80):
    middle_tech_scenario[i] = middle_tech_scenario[i-1] + (80 - i)/80
for model_id in range(0,18):
    print(model_id)
    almond_hist = nc.Dataset(input_path+str(model_list[model_id])+'_hist_ACI.nc')
    almond_rcp45 = nc.Dataset(input_path+str(model_list[model_id])+'_rcp45_ACI.nc')
    almond_rcp85 = nc.Dataset(input_path+str(model_list[model_id])+'_rcp85_ACI.nc')
    ACI_rcp45_1980_2020_sum = np.zeros((0,aci_num*2+32))
    ACI_rcp45_s_1980_2020_sum = np.zeros((0,aci_num*2+32))
    ACI_rcp45_m_1980_2020_sum = np.zeros((0,aci_num*2+32))
    ACI_rcp85_1980_2020_sum = np.zeros((0,aci_num*2+32))
    ACI_rcp85_s_1980_2020_sum = np.zeros((0,aci_num*2+32))
    ACI_rcp85_m_1980_2020_sum = np.zeros((0,aci_num*2+32))
    ACI_rcp45_2021_2099_sum = np.zeros((0,aci_num*2+32))
    ACI_rcp45_s_2021_2099_sum = np.zeros((0,aci_num*2+32))
    ACI_rcp45_m_2021_2099_sum = np.zeros((0,aci_num*2+32))
    ACI_rcp85_2021_2099_sum = np.zeros((0,aci_num*2+32))
    ACI_rcp85_s_2021_2099_sum = np.zeros((0,aci_num*2+32))
    ACI_rcp85_m_2021_2099_sum = np.zeros((0,aci_num*2+32))
    aci_sum_state_hist = np.zeros((16*26, aci_num))
    aci_sum_state_rcp45 = np.zeros((16*15, aci_num))
    aci_sum_state_rcp85 = np.zeros((16*15, aci_num))
    for i in range(aci_num):
        aci_sum_state_hist[:,i] = np.ndarray.flatten(almond_hist.variables['ACI_value'][29:55,:,i])
        aci_sum_state_rcp45[:,i] = np.ndarray.flatten(almond_rcp45.variables['ACI_value'][0:15,:,i])
        aci_sum_state_rcp85[:,i] = np.ndarray.flatten(almond_rcp85.variables['ACI_value'][0:15,:,i])
    aci_sum_state_hist_rcp45 = np.row_stack((aci_sum_state_hist, aci_sum_state_rcp45))
    aci_sum_state_hist_rcp85 = np.row_stack((aci_sum_state_hist, aci_sum_state_rcp85))
    aci_sum_state_hist_rcp45_sq = aci_sum_state_hist_rcp45**2
    aci_sum_state_hist_rcp85_sq = aci_sum_state_hist_rcp85**2
    aci_mean_rcp45 = np.mean(aci_sum_state_hist_rcp45, axis = 0)
    aci_mean_rcp85 = np.mean(aci_sum_state_hist_rcp85, axis = 0)
    aci_std_rcp45 = np.std(aci_sum_state_hist_rcp45, axis = 0)
    aci_std_rcp85 = np.std(aci_sum_state_hist_rcp85, axis = 0)
    aci_mean_rcp45_sq = np.mean(aci_sum_state_hist_rcp45_sq, axis=0)
    aci_mean_rcp85_sq = np.mean(aci_sum_state_hist_rcp85_sq, axis=0)
    aci_std_rcp45_sq = np.std(aci_sum_state_hist_rcp45_sq, axis=0)
    aci_std_rcp85_sq = np.std(aci_sum_state_hist_rcp85_sq, axis=0)
    for region_id in range(0,16):
        print(region_id)
        non_clim_coef_hist = np.zeros((26,32))
        non_clim_coef_hist[:,region_id] = np.arange(1,27,1)
        non_clim_coef_hist[:,(region_id+16)] = 1
        aci_hist = np.array(almond_hist.variables['ACI_value'][29:55,region_id,:], dtype = np.float64)
        aci_hist = np.column_stack((aci_hist, aci_hist**2))
        aci_rcp45 = np.array(almond_rcp45.variables['ACI_value'][:,region_id,:], dtype = np.float64)
        aci_rcp45 = np.column_stack((aci_rcp45, aci_rcp45**2))
        ACI_rcp85 = np.array(almond_rcp85.variables['ACI_value'][:,region_id,:], dtype = np.float64)
        ACI_rcp85 = np.column_stack((ACI_rcp85, ACI_rcp85**2))
        non_clim_coef= np.zeros((94,32))
        non_clim_coef[:,region_id] = np.arange(27,121,1)
        non_clim_coef[:,(region_id+16)] = 1
        non_clim_coef_s = np.zeros((94, 32))
        non_clim_coef_s[0:15,region_id] = np.arange(27,42,1)
        non_clim_coef_s[15:,region_id] = 41
        non_clim_coef_s[:,(region_id+16)] = 1
        non_clim_coef_m = np.zeros((94, 32))
        non_clim_coef_m[0:15,region_id] = np.arange(27,42,1)
        non_clim_coef_m[15:,region_id] = middle_tech_scenario[1:]
        non_clim_coef_m[:,(region_id+16)] = 1
        ACI_rcp45_1980_2020 = np.column_stack((np.row_stack((aci_hist, aci_rcp45[0:15])), np.row_stack((non_clim_coef_hist, non_clim_coef[0:15]))))
        ACI_rcp45_s_1980_2020 = np.column_stack((np.row_stack((aci_hist, aci_rcp45[0:15])), np.row_stack((non_clim_coef_hist, non_clim_coef_s[0:15]))))
        ACI_rcp45_m_1980_2020 = np.column_stack((np.row_stack((aci_hist, aci_rcp45[0:15])), np.row_stack((non_clim_coef_hist, non_clim_coef_m[0:15]))))
        ACI_rcp85_1980_2020 = np.column_stack((np.row_stack((aci_hist, ACI_rcp85[0:15])), np.row_stack((non_clim_coef_hist, non_clim_coef[0:15]))))
        ACI_rcp85_s_1980_2020 = np.column_stack((np.row_stack((aci_hist, ACI_rcp85[0:15])), np.row_stack((non_clim_coef_hist, non_clim_coef_s[0:15]))))
        ACI_rcp85_m_1980_2020 = np.column_stack((np.row_stack((aci_hist, ACI_rcp85[0:15])), np.row_stack((non_clim_coef_hist, non_clim_coef_m[0:15]))))
        ACI_rcp45_2021_2099 = np.column_stack((aci_rcp45[15:94], non_clim_coef[15:94]))
        ACI_rcp45_s_2021_2099 = np.column_stack((aci_rcp45[15:94], non_clim_coef_s[15:94]))
        ACI_rcp45_m_2021_2099 = np.column_stack((aci_rcp45[15:94], non_clim_coef_m[15:94]))
        ACI_rcp85_2021_2099 = np.column_stack((ACI_rcp85[15:94], non_clim_coef[15:94]))
        ACI_rcp85_s_2021_2099 = np.column_stack((ACI_rcp85[15:94], non_clim_coef_s[15:94]))
        ACI_rcp85_m_2021_2099 = np.column_stack((ACI_rcp85[15:94], non_clim_coef_m[15:94]))
        for j in range(0,aci_num):
            ACI_rcp45_2021_2099[:,j] = (ACI_rcp45_2021_2099[:,j]-aci_mean_rcp45[j])/aci_std_rcp45[j]
            ACI_rcp45_s_2021_2099[:,j] = (ACI_rcp45_s_2021_2099[:,j]-aci_mean_rcp45[j])/aci_std_rcp45[j]
            ACI_rcp45_m_2021_2099[:,j] = (ACI_rcp45_m_2021_2099[:,j]-aci_mean_rcp45[j])/aci_std_rcp45[j]
            ACI_rcp45_1980_2020[:,j] = (ACI_rcp45_1980_2020[:,j]-aci_mean_rcp45[j])/aci_std_rcp45[j]
            ACI_rcp45_s_1980_2020[:,j] = (ACI_rcp45_s_1980_2020[:,j]-aci_mean_rcp45[j])/aci_std_rcp45[j]
            ACI_rcp45_m_1980_2020[:,j] = (ACI_rcp45_m_1980_2020[:,j]-aci_mean_rcp45[j])/aci_std_rcp45[j]
            ACI_rcp85_2021_2099[:,j] = (ACI_rcp85_2021_2099[:,j]-aci_mean_rcp85[j])/aci_std_rcp85[j]
            ACI_rcp85_s_2021_2099[:,j] = (ACI_rcp85_s_2021_2099[:,j]-aci_mean_rcp85[j])/aci_std_rcp85[j]
            ACI_rcp85_m_2021_2099[:,j] = (ACI_rcp85_m_2021_2099[:,j]-aci_mean_rcp85[j])/aci_std_rcp85[j]
            ACI_rcp85_1980_2020[:,j] = (ACI_rcp85_1980_2020[:,j]-aci_mean_rcp85[j])/aci_std_rcp85[j]
            ACI_rcp85_s_1980_2020[:,j] = (ACI_rcp85_s_1980_2020[:,j]-aci_mean_rcp85[j])/aci_std_rcp85[j]
            ACI_rcp85_m_1980_2020[:,j] = (ACI_rcp85_m_1980_2020[:,j]-aci_mean_rcp85[j])/aci_std_rcp85[j]
            ACI_rcp45_2021_2099[:,j+aci_num] = (ACI_rcp45_2021_2099[:,j+aci_num]-aci_mean_rcp45_sq[j])/aci_std_rcp45_sq[j]
            ACI_rcp45_s_2021_2099[:,j+aci_num] = (ACI_rcp45_s_2021_2099[:,j+aci_num]-aci_mean_rcp45_sq[j])/aci_std_rcp45_sq[j]
            ACI_rcp45_m_2021_2099[:,j+aci_num] = (ACI_rcp45_m_2021_2099[:,j+aci_num]-aci_mean_rcp45_sq[j])/aci_std_rcp45_sq[j]
            ACI_rcp45_1980_2020[:,j+aci_num] = (ACI_rcp45_1980_2020[:,j+aci_num]-aci_mean_rcp45_sq[j])/aci_std_rcp45_sq[j]
            ACI_rcp45_s_1980_2020[:,j+aci_num] = (ACI_rcp45_s_1980_2020[:,j+aci_num]-aci_mean_rcp45_sq[j])/aci_std_rcp45_sq[j]
            ACI_rcp45_m_1980_2020[:,j+aci_num] = (ACI_rcp45_m_1980_2020[:,j+aci_num]-aci_mean_rcp45_sq[j])/aci_std_rcp45_sq[j]
            ACI_rcp85_2021_2099[:,j+aci_num] = (ACI_rcp85_2021_2099[:,j+aci_num]-aci_mean_rcp85_sq[j])/aci_std_rcp85_sq[j]
            ACI_rcp85_s_2021_2099[:,j+aci_num] = (ACI_rcp85_s_2021_2099[:,j+aci_num]-aci_mean_rcp85_sq[j])/aci_std_rcp85_sq[j]
            ACI_rcp85_m_2021_2099[:,j+aci_num] = (ACI_rcp85_m_2021_2099[:,j+aci_num]-aci_mean_rcp85_sq[j])/aci_std_rcp85_sq[j]
            ACI_rcp85_1980_2020[:,j+aci_num] = (ACI_rcp85_1980_2020[:,j+aci_num]-aci_mean_rcp85_sq[j])/aci_std_rcp85_sq[j]
            ACI_rcp85_s_1980_2020[:,j+aci_num] = (ACI_rcp85_s_1980_2020[:,j+aci_num]-aci_mean_rcp85_sq[j])/aci_std_rcp85_sq[j]
            ACI_rcp85_m_1980_2020[:,j+aci_num] = (ACI_rcp85_m_1980_2020[:,j+aci_num]-aci_mean_rcp85_sq[j])/aci_std_rcp85_sq[j]
        ACI_rcp45_1980_2020_sum = np.row_stack((ACI_rcp45_1980_2020_sum, ACI_rcp45_1980_2020))
        ACI_rcp45_s_1980_2020_sum = np.row_stack((ACI_rcp45_s_1980_2020_sum,ACI_rcp45_s_1980_2020))
        ACI_rcp45_m_1980_2020_sum = np.row_stack((ACI_rcp45_m_1980_2020_sum,ACI_rcp45_m_1980_2020))
        ACI_rcp85_1980_2020_sum = np.row_stack((ACI_rcp85_1980_2020_sum,ACI_rcp85_1980_2020))
        ACI_rcp85_s_1980_2020_sum = np.row_stack((ACI_rcp85_s_1980_2020_sum,ACI_rcp85_s_1980_2020))
        ACI_rcp85_m_1980_2020_sum = np.row_stack((ACI_rcp85_m_1980_2020_sum,ACI_rcp85_m_1980_2020))
        ACI_rcp45_2021_2099_sum = np.row_stack((ACI_rcp45_2021_2099_sum,ACI_rcp45_2021_2099))
        ACI_rcp45_s_2021_2099_sum = np.row_stack((ACI_rcp45_s_2021_2099_sum,ACI_rcp45_s_2021_2099))
        ACI_rcp45_m_2021_2099_sum = np.row_stack((ACI_rcp45_m_2021_2099_sum,ACI_rcp45_m_2021_2099))
        ACI_rcp85_2021_2099_sum = np.row_stack((ACI_rcp85_2021_2099_sum,ACI_rcp85_2021_2099))
        ACI_rcp85_s_2021_2099_sum = np.row_stack((ACI_rcp85_s_2021_2099_sum,ACI_rcp85_s_2021_2099))
        ACI_rcp85_m_2021_2099_sum = np.row_stack((ACI_rcp85_m_2021_2099_sum,ACI_rcp85_m_2021_2099))
        print(non_clim_coef_m)
        print(middle_tech_scenario)
    savetxt(str(save_path)+str(model_list[model_id])+'hist_rcp45_ACI.csv', ACI_rcp45_1980_2020_sum, delimiter = ',')
    savetxt(str(save_path)+str(model_list[model_id])+'hist_rcp45_s_ACI.csv', ACI_rcp45_s_1980_2020_sum, delimiter = ',')
    savetxt(str(save_path)+str(model_list[model_id])+'hist_rcp45_m_ACI.csv', ACI_rcp45_m_1980_2020_sum, delimiter = ',')
    savetxt(str(save_path)+str(model_list[model_id])+'hist_rcp85_ACI.csv', ACI_rcp85_1980_2020_sum, delimiter = ',')
    savetxt(str(save_path)+str(model_list[model_id])+'hist_rcp85_s_ACI.csv', ACI_rcp85_s_1980_2020_sum, delimiter = ',')
    savetxt(str(save_path)+str(model_list[model_id])+'hist_rcp85_m_ACI.csv', ACI_rcp85_m_1980_2020_sum, delimiter = ',')
    savetxt(str(save_path)+str(model_list[model_id])+'future_rcp45_ACI.csv', ACI_rcp45_2021_2099_sum, delimiter = ',')
    savetxt(str(save_path)+str(model_list[model_id])+'future_rcp45_s_ACI.csv', ACI_rcp45_s_2021_2099_sum, delimiter = ',')
    savetxt(str(save_path)+str(model_list[model_id])+'future_rcp45_m_ACI.csv', ACI_rcp45_m_2021_2099_sum, delimiter = ',')
    savetxt(str(save_path)+str(model_list[model_id])+'future_rcp85_ACI.csv', ACI_rcp85_2021_2099_sum, delimiter = ',')
    savetxt(str(save_path)+str(model_list[model_id])+'future_rcp85_s_ACI.csv', ACI_rcp85_s_2021_2099_sum, delimiter = ',')
    savetxt(str(save_path)+str(model_list[model_id])+'future_rcp85_m_ACI.csv', ACI_rcp85_m_2021_2099_sum, delimiter = ',')

