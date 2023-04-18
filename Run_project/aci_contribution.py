import math 
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import netCDF4 as nc
from numpy import genfromtxt
import scipy.stats as st
from numpy import savetxt
import matplotlib.patches as mpatches
import matplotlib
import seaborn as sns
from scipy import signal
from scipy import stats
from matplotlib.lines import Line2D
import matplotlib as mpl


aci_num = 13
data_ID='11_19'
load_coef_path = '/home/shqwu/Almond_code_git/saved_data/'+str(data_ID)+'/lasso_model/'
load_aci_path = '/home/shqwu/Almond_code_git/saved_data/'+str(data_ID)+'/MACA_csv/to_2020/'
area_csv = genfromtxt('/home/shqwu/Almond_code_git/almond_area.csv', delimiter = ',')
save_path = '/home/shqwu/Almond_code_git/saved_data/'+str(data_ID)+'/aci_contribution/'
##multiply coef and aci
for i in range(1,11):
    locals()['coef'+str(i)] = np.zeros((100,aci_num*2+32))
coef_sum = np.zeros((0,aci_num*2+32))
for i in range(1,1001):
    locals()['coef'+str(((i-1)//100)+1)][i%100-1] = genfromtxt(str(load_coef_path) + 'coef_'+str(i)+'.csv', delimiter = ',')
for i in range(1,11):
    coef_sum = np.row_stack((coef_sum, locals()['coef'+str(i)]))

model_list = ['bcc-csm1-1-m', 'bcc-csm1-1','BNU-ESM', 'CanESM2', 'CSIRO-Mk3-6-0', 'GFDL-ESM2G', 'GFDL-ESM2M', 'inmcm4', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR','CNRM-CM5', 'HadGEM2-CC365','HadGEM2-ES365', 'IPSL-CM5B-LR', 'MIROC5', 'MIROC-ESM', 'MIROC-ESM-CHEM']

## obtain aci*coef area-weighted CA
for model in range(0,17):
    locals()['aci_contribution_model_rcp45_'+str(model)] = np.zeros((656,1000,aci_num*2))
    locals()['aci_contribution_model_rcp85_'+str(model)] = np.zeros((656,1000,aci_num*2))
    for trial in range(1,11):
        simulation_rcp45 = np.zeros((656,100))
        simulation_rcp85 = np.zeros((656,100))
        aci_rcp45 = genfromtxt(str(load_aci_path)+str(model_list[model])+'hist_rcp45_ACI.csv', delimiter = ',')[:,0:aci_num*2]
        aci_rcp85 = genfromtxt(str(load_aci_path)+str(model_list[model])+'hist_rcp85_ACI.csv', delimiter = ',')[:,0:aci_num*2]
        for i in range(0,656):
            for j in range(0,100):
                locals()['aci_contribution_model_rcp45_'+str(model)][i,(trial-1)*100+j,:] = aci_rcp45[i,:]*locals()['coef'+str(trial)][j,0:aci_num*2]
                locals()['aci_contribution_model_rcp85_'+str(model)][i,(trial-1)*100+j,:] = aci_rcp85[i,:]*locals()['coef'+str(trial)][j,0:aci_num*2]
area = area_csv[0:41,:]
for model in range(0,17):
    for index in range(0,16):
        for year in range(0,41):
            locals()['aci_contribution_model_rcp45_'+str(model)][index*41+year,:,:] = locals()['aci_contribution_model_rcp45_'+str(model)][index*41+year,:,:]*area[year,index]/np.sum(area[year])             
            locals()['aci_contribution_model_rcp85_'+str(model)][index*41+year,:,:] = locals()['aci_contribution_model_rcp85_'+str(model)][index*41+year,:,:]*area[year,index]/np.sum(area[year])
for model in range(0,17):
    locals()['aci_contribution_model_rcp45_sum'+str(model)] = np.zeros((41,1000,aci_num*2))
    locals()['aci_contribution_model_rcp85_sum'+str(model)] = np.zeros((41,1000,aci_num*2))
    for county_id in range(0,16):
        for year in range(0,41):
            locals()['aci_contribution_model_rcp45_sum'+str(model)][year,:,:] = locals()['aci_contribution_model_rcp45_sum'+str(model)][year,:,:] + np.split(locals()['aci_contribution_model_rcp45_'+str(model)],16,axis=0)[county_id][year,:,:]      
            locals()['aci_contribution_model_rcp85_sum'+str(model)][year,:,:] = locals()['aci_contribution_model_rcp85_sum'+str(model)][year,:,:] + np.split(locals()['aci_contribution_model_rcp85_'+str(model)],16,axis=0)[county_id][year,:,:]      
aci_contribution_model_rcp45 = np.zeros((17,41,1000,aci_num*2))
aci_contribution_model_rcp85 = np.zeros((17,41,1000,aci_num*2))
for model in range(0,17):
    aci_contribution_model_rcp45[model, :,:,:] = locals()['aci_contribution_model_rcp45_sum'+str(model)]
    aci_contribution_model_rcp85[model, :,:,:] = locals()['aci_contribution_model_rcp85_sum'+str(model)]
aci_contribution_model_rcp45 = np.mean(aci_contribution_model_rcp45, axis=0)
aci_contribution_model_rcp85 = np.mean(aci_contribution_model_rcp85, axis=0)

for model in range(0,17):
    locals()['aci_contribution_model_rcp45_future_'+str(model)] = np.zeros((1264,1000,aci_num*2))
    locals()['aci_contribution_model_rcp85_future_'+str(model)] = np.zeros((1264,1000,aci_num*2))
    for trial in range(1,11):
        simulation_rcp45 = np.zeros((1264,100))
        simulation_rcp85 = np.zeros((1264,100))
        aci_rcp45 = genfromtxt(str(load_aci_path)+str(model_list[model])+'future_rcp45_ACI.csv', delimiter = ',')[:,0:aci_num*2]
        aci_rcp85 = genfromtxt(str(load_aci_path)+str(model_list[model])+'future_rcp85_ACI.csv', delimiter = ',')[:,0:aci_num*2]
        for i in range(0,1264):
            for j in range(0,100):
                locals()['aci_contribution_model_rcp45_future_'+str(model)][i,(trial-1)*100+j,:] = aci_rcp45[i,:]*locals()['coef'+str(trial)][j,0:aci_num*2]
                locals()['aci_contribution_model_rcp85_future_'+str(model)][i,(trial-1)*100+j,:] = aci_rcp85[i,:]*locals()['coef'+str(trial)][j,0:aci_num*2]
for model in range(0,17):
    for index in range(0,16):
        locals()['aci_contribution_model_rcp45_future_'+str(model)][index*79:(index+1)*79,:,:] = locals()['aci_contribution_model_rcp45_future_'+str(model)][index*79:(index+1)*79,:,:]*area[-1,index]/np.sum(area[-1])             
        locals()['aci_contribution_model_rcp85_future_'+str(model)][index*79:(index+1)*79,:,:] = locals()['aci_contribution_model_rcp85_future_'+str(model)][index*79:(index+1)*79,:,:]*area[-1,index]/np.sum(area[-1])
for model in range(0,17):
    locals()['aci_contribution_model_rcp45_sum_future_'+str(model)] = np.zeros((79,1000,aci_num*2))
    locals()['aci_contribution_model_rcp85_sum_future_'+str(model)] = np.zeros((79,1000,aci_num*2))
    for county_id in range(0,16):
        for year in range(0,79):
            locals()['aci_contribution_model_rcp45_sum_future_'+str(model)][year,:,:] = locals()['aci_contribution_model_rcp45_sum_future_'+str(model)][year,:,:] + np.split(locals()['aci_contribution_model_rcp45_future_'+str(model)],16,axis=0)[county_id][year,:,:]      
            locals()['aci_contribution_model_rcp85_sum_future_'+str(model)][year,:,:] = locals()['aci_contribution_model_rcp85_sum_future_'+str(model)][year,:,:] + np.split(locals()['aci_contribution_model_rcp85_future_'+str(model)],16,axis=0)[county_id][year,:,:]      
aci_contribution_model_rcp45_future = np.zeros((17,79,1000,aci_num*2))
aci_contribution_model_rcp85_future = np.zeros((17,79,1000,aci_num*2))
for model in range(0,17):
    aci_contribution_model_rcp45_future[model, :,:,:] = locals()['aci_contribution_model_rcp45_sum_future_'+str(model)]
    aci_contribution_model_rcp85_future[model, :,:,:] = locals()['aci_contribution_model_rcp85_sum_future_'+str(model)]
aci_contribution_model_rcp45_future = np.mean(aci_contribution_model_rcp45_future, axis=0)
aci_contribution_model_rcp85_future = np.mean(aci_contribution_model_rcp85_future, axis=0)

aci_contribution_rcp45_total = np.row_stack((aci_contribution_model_rcp45, aci_contribution_model_rcp45_future))
aci_contribution_rcp85_total = np.row_stack((aci_contribution_model_rcp85, aci_contribution_model_rcp85_future))

for i in range(0,aci_num):
    aci_contribution_rcp45_total[:,:,i] = aci_contribution_rcp45_total[:,:,i]+aci_contribution_rcp45_total[:,:,i+aci_num]
    aci_contribution_rcp85_total[:,:,i] = aci_contribution_rcp85_total[:,:,i]+aci_contribution_rcp85_total[:,:,i+aci_num]

aci_contribution_rcp45_total = aci_contribution_rcp45_total[:,:,0:aci_num]
aci_contribution_rcp85_total = aci_contribution_rcp85_total[:,:,0:aci_num]


aci_contribution_rcp45_county_hist = np.zeros((16,17,41,1000,aci_num))
aci_contribution_rcp85_county_hist = np.zeros((16,17,41,1000,aci_num))
aci_contribution_rcp45_county_future = np.zeros((16,17,79,1000,aci_num))
aci_contribution_rcp85_county_future = np.zeros((16,17,79,1000,aci_num))
for county in range(0,16):
    for model in range(0,17):
        aci_contribution_rcp45_county_hist[county,model,:,:,:] = np.split(locals()['aci_contribution_model_rcp45_'+str(model)],16)[county][:,:,0:aci_num]+np.split(locals()['aci_contribution_model_rcp45_'+str(model)],16)[county][:,:,aci_num:aci_num*2]
        aci_contribution_rcp85_county_hist[county,model,:,:,:] = np.split(locals()['aci_contribution_model_rcp85_'+str(model)],16)[county][:,:,0:aci_num]+np.split(locals()['aci_contribution_model_rcp85_'+str(model)],16)[county][:,:,aci_num:aci_num*2]
        aci_contribution_rcp45_county_future[county,model,:,:,:] = np.split(locals()['aci_contribution_model_rcp45_future_'+str(model)],16)[county][:,:,0:aci_num]+np.split(locals()['aci_contribution_model_rcp45_future_'+str(model)],16)[county][:,:,aci_num:aci_num*2]
        aci_contribution_rcp85_county_future[county,model,:,:,:] = np.split(locals()['aci_contribution_model_rcp85_future_'+str(model)],16)[county][:,:,0:aci_num]+np.split(locals()['aci_contribution_model_rcp85_future_'+str(model)],16)[county][:,:,aci_num:aci_num*2]
aci_contribution_rcp45_county = np.concatenate((aci_contribution_rcp45_county_hist, aci_contribution_rcp45_county_future),axis=2)
aci_contribution_rcp85_county = np.concatenate((aci_contribution_rcp85_county_hist, aci_contribution_rcp85_county_future),axis=2)

aci_contribution_rcp45_county_2050_change =  np.mean((np.mean(aci_contribution_rcp45_county[:,:,60:80,:,:], axis=2)-np.mean(aci_contribution_rcp45_county[:,:,21:41,:,:],axis=2)), axis=1)
aci_contribution_rcp45_county_2090_change =  np.mean((np.mean(aci_contribution_rcp45_county[:,:,100:120,:,:], axis=2)-np.mean(aci_contribution_rcp45_county[:,:,21:41,:,:],axis=2)), axis=1)
aci_contribution_rcp45_county_2050_change_percent = np.zeros((16,1000,aci_num))
aci_contribution_rcp45_county_2090_change_percent = np.zeros((16,1000,aci_num))

aci_contribution_rcp85_county_2050_change =  np.mean((np.mean(aci_contribution_rcp85_county[:,:,60:80,:,:], axis=2)-np.mean(aci_contribution_rcp85_county[:,:,21:41,:,:],axis=2)), axis=1)
aci_contribution_rcp85_county_2090_change =  np.mean((np.mean(aci_contribution_rcp85_county[:,:,100:120,:,:], axis=2)-np.mean(aci_contribution_rcp85_county[:,:,21:41,:,:],axis=2)), axis=1)
aci_contribution_rcp85_county_2050_change_percent = np.zeros((16,1000,aci_num))
aci_contribution_rcp85_county_2090_change_percent = np.zeros((16,1000,aci_num))

for county in range(0,16):
    for trial in range(0,1000):
        aci_contribution_rcp45_county_2050_change_percent[county,trial,:] = 100*aci_contribution_rcp45_county_2050_change[county,trial,:]/np.abs(np.sum(aci_contribution_rcp45_county_2050_change[county,trial,:]))
        aci_contribution_rcp45_county_2090_change_percent[county,trial,:] = 100*aci_contribution_rcp45_county_2090_change[county,trial,:]/np.abs(np.sum(aci_contribution_rcp45_county_2090_change[county,trial,:]))
        aci_contribution_rcp85_county_2050_change_percent[county,trial,:] = 100*aci_contribution_rcp85_county_2050_change[county,trial,:]/np.abs(np.sum(aci_contribution_rcp85_county_2050_change[county,trial,:]))
        aci_contribution_rcp85_county_2090_change_percent[county,trial,:] = 100*aci_contribution_rcp85_county_2090_change[county,trial,:]/np.abs(np.sum(aci_contribution_rcp85_county_2090_change[county,trial,:]))

np.save(str(save_path)+'aci_contribution_rcp45_total.npy',aci_contribution_rcp45_total)
np.save(str(save_path)+'aci_contribution_rcp85_total.npy',aci_contribution_rcp85_total)

np.save(str(save_path)+'aci_contribution_rcp45_county_2050_change_percent.npy',aci_contribution_rcp45_county_2050_change_percent)
np.save(str(save_path)+'aci_contribution_rcp85_county_2050_change_percent.npy',aci_contribution_rcp85_county_2050_change_percent)
np.save(str(save_path)+'aci_contribution_rcp45_county_2090_change_percent.npy',aci_contribution_rcp45_county_2090_change_percent)
np.save(str(save_path)+'aci_contribution_rcp85_county_2090_change_percent.npy',aci_contribution_rcp85_county_2090_change_percent)

np.save(str(save_path)+'aci_contribution_rcp45_county_2050_change.npy',aci_contribution_rcp45_county_2050_change)
np.save(str(save_path)+'aci_contribution_rcp85_county_2050_change.npy',aci_contribution_rcp85_county_2050_change)
np.save(str(save_path)+'aci_contribution_rcp45_county_2090_change.npy',aci_contribution_rcp45_county_2090_change)
np.save(str(save_path)+'aci_contribution_rcp85_county_2090_change.npy',aci_contribution_rcp85_county_2090_change)



