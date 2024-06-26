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


aci_num = 14
home_path='~/Run_project'
input_path_model = home_path+'/intermediate_data/lasso_model/'
input_path_ACI = home_path+'/intermediate_data/LOCA_csv/'
input_path_area = home_path+'/input_data/'
save_path = home_path+'/output_data/aci_contribution/LOCA/'


area_csv = genfromtxt(input_path_area+'almond_area.csv', delimiter = ',')


##multiply coef and aci
for i in range(1,11):
    locals()['coef'+str(i)] = np.zeros((100,aci_num*2+32))
coef_sum = np.zeros((0,aci_num*2+32))
for i in range(1,1001):
    locals()['coef'+str(((i-1)//100)+1)][i%100-1] = genfromtxt(input_path_model+'coef_'+str(i)+'.csv', delimiter = ',')
for i in range(1,11):
    coef_sum = np.row_stack((coef_sum, locals()['coef'+str(i)]))

model_list = ['ACCESS-CM2', 'EC-Earth3', 'EC-Earth3-Veg', 'GFDL-ESM4',  'INM-CM5-0',  'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'CNRM-ESM2-1']


## obtain aci*coef area-weighted CA
for model in range(0,8):
    locals()['aci_contribution_model_ssp245_'+str(model)] = np.zeros((656,1000,aci_num*2))
    locals()['aci_contribution_model_ssp585_'+str(model)] = np.zeros((656,1000,aci_num*2))
    for trial in range(1,11):
        simulation_ssp245 = np.zeros((656,100))
        simulation_ssp585 = np.zeros((656,100))
        aci_ssp245 = genfromtxt(input_path_ACI+str(model_list[model])+'hist_ssp245_ACI.csv', delimiter = ',')[:,0:aci_num*2]
        aci_ssp585 = genfromtxt(input_path_ACI+str(model_list[model])+'hist_ssp585_ACI.csv', delimiter = ',')[:,0:aci_num*2]
        for i in range(0,656):
            for j in range(0,100):
                locals()['aci_contribution_model_ssp245_'+str(model)][i,(trial-1)*100+j,:] = aci_ssp245[i,:]*locals()['coef'+str(trial)][j,0:aci_num*2]
                locals()['aci_contribution_model_ssp585_'+str(model)][i,(trial-1)*100+j,:] = aci_ssp585[i,:]*locals()['coef'+str(trial)][j,0:aci_num*2]
    locals()['aci_contribution_model_ssp245_copy_'+str(model)] = np.copy(locals()['aci_contribution_model_ssp245_'+str(model)])
    locals()['aci_contribution_model_ssp585_copy_'+str(model)] = np.copy(locals()['aci_contribution_model_ssp585_'+str(model)])

area = area_csv[0:41,:]
for model in range(0,8):
    for index in range(0,16):
        for year in range(0,41):
            locals()['aci_contribution_model_ssp245_'+str(model)][index*41+year,:,:] = locals()['aci_contribution_model_ssp245_'+str(model)][index*41+year,:,:]*area[year,index]/np.sum(area[year])             
            locals()['aci_contribution_model_ssp585_'+str(model)][index*41+year,:,:] = locals()['aci_contribution_model_ssp585_'+str(model)][index*41+year,:,:]*area[year,index]/np.sum(area[year])
for model in range(0,8):
    locals()['aci_contribution_model_ssp245_sum'+str(model)] = np.zeros((41,1000,aci_num*2))
    locals()['aci_contribution_model_ssp585_sum'+str(model)] = np.zeros((41,1000,aci_num*2))
    for county_id in range(0,16):
        for year in range(0,41):
            locals()['aci_contribution_model_ssp245_sum'+str(model)][year,:,:] = locals()['aci_contribution_model_ssp245_sum'+str(model)][year,:,:] + np.split(locals()['aci_contribution_model_ssp245_'+str(model)],16,axis=0)[county_id][year,:,:]      
            locals()['aci_contribution_model_ssp585_sum'+str(model)][year,:,:] = locals()['aci_contribution_model_ssp585_sum'+str(model)][year,:,:] + np.split(locals()['aci_contribution_model_ssp585_'+str(model)],16,axis=0)[county_id][year,:,:]      
aci_contribution_model_ssp245 = np.zeros((8,41,1000,aci_num*2))
aci_contribution_model_ssp585 = np.zeros((8,41,1000,aci_num*2))
for model in range(0,8):
    aci_contribution_model_ssp245[model, :,:,:] = locals()['aci_contribution_model_ssp245_sum'+str(model)]
    aci_contribution_model_ssp585[model, :,:,:] = locals()['aci_contribution_model_ssp585_sum'+str(model)]
aci_contribution_model_ssp245 = np.mean(aci_contribution_model_ssp245, axis=0)
aci_contribution_model_ssp585 = np.mean(aci_contribution_model_ssp585, axis=0)

for model in range(0,8):
    locals()['aci_contribution_model_ssp245_future_'+str(model)] = np.zeros((1264,1000,aci_num*2))
    locals()['aci_contribution_model_ssp585_future_'+str(model)] = np.zeros((1264,1000,aci_num*2))
    for trial in range(1,11):
        simulation_ssp245 = np.zeros((1264,100))
        simulation_ssp585 = np.zeros((1264,100))
        aci_ssp245 = genfromtxt(input_path_ACI+str(model_list[model])+'future_ssp245_ACI.csv', delimiter = ',')[:,0:aci_num*2]
        aci_ssp585 = genfromtxt(input_path_ACI+str(model_list[model])+'future_ssp585_ACI.csv', delimiter = ',')[:,0:aci_num*2]
        for i in range(0,1264):
            for j in range(0,100):
                locals()['aci_contribution_model_ssp245_future_'+str(model)][i,(trial-1)*100+j,:] = aci_ssp245[i,:]*locals()['coef'+str(trial)][j,0:aci_num*2]
                locals()['aci_contribution_model_ssp585_future_'+str(model)][i,(trial-1)*100+j,:] = aci_ssp585[i,:]*locals()['coef'+str(trial)][j,0:aci_num*2]
    locals()['aci_contribution_model_ssp245_future_copy_'+str(model)] = np.copy(locals()['aci_contribution_model_ssp245_future_'+str(model)])
    locals()['aci_contribution_model_ssp585_future_copy_'+str(model)] = np.copy(locals()['aci_contribution_model_ssp585_future_'+str(model)])
    
for model in range(0,8):
    for index in range(0,16):
        locals()['aci_contribution_model_ssp245_future_'+str(model)][index*79:(index+1)*79,:,:] = locals()['aci_contribution_model_ssp245_future_'+str(model)][index*79:(index+1)*79,:,:]*area[-1,index]/np.sum(area[-1])             
        locals()['aci_contribution_model_ssp585_future_'+str(model)][index*79:(index+1)*79,:,:] = locals()['aci_contribution_model_ssp585_future_'+str(model)][index*79:(index+1)*79,:,:]*area[-1,index]/np.sum(area[-1])
for model in range(0,8):
    locals()['aci_contribution_model_ssp245_sum_future_'+str(model)] = np.zeros((79,1000,aci_num*2))
    locals()['aci_contribution_model_ssp585_sum_future_'+str(model)] = np.zeros((79,1000,aci_num*2))
    for county_id in range(0,16):
        for year in range(0,79):
            locals()['aci_contribution_model_ssp245_sum_future_'+str(model)][year,:,:] = locals()['aci_contribution_model_ssp245_sum_future_'+str(model)][year,:,:] + np.split(locals()['aci_contribution_model_ssp245_future_'+str(model)],16,axis=0)[county_id][year,:,:]      
            locals()['aci_contribution_model_ssp585_sum_future_'+str(model)][year,:,:] = locals()['aci_contribution_model_ssp585_sum_future_'+str(model)][year,:,:] + np.split(locals()['aci_contribution_model_ssp585_future_'+str(model)],16,axis=0)[county_id][year,:,:]      
aci_contribution_model_ssp245_future = np.zeros((8,79,1000,aci_num*2))
aci_contribution_model_ssp585_future = np.zeros((8,79,1000,aci_num*2))
for model in range(0,8):
    aci_contribution_model_ssp245_future[model, :,:,:] = locals()['aci_contribution_model_ssp245_sum_future_'+str(model)]
    aci_contribution_model_ssp585_future[model, :,:,:] = locals()['aci_contribution_model_ssp585_sum_future_'+str(model)]
aci_contribution_model_ssp245_future = np.mean(aci_contribution_model_ssp245_future, axis=0)
aci_contribution_model_ssp585_future = np.mean(aci_contribution_model_ssp585_future, axis=0)

aci_contribution_ssp245_total = np.row_stack((aci_contribution_model_ssp245, aci_contribution_model_ssp245_future))
aci_contribution_ssp585_total = np.row_stack((aci_contribution_model_ssp585, aci_contribution_model_ssp585_future))
for i in range(0,aci_num):
    aci_contribution_ssp245_total[:,:,i] = aci_contribution_ssp245_total[:,:,i]+aci_contribution_ssp245_total[:,:,i+aci_num]
    aci_contribution_ssp585_total[:,:,i] = aci_contribution_ssp585_total[:,:,i]+aci_contribution_ssp585_total[:,:,i+aci_num]

aci_contribution_ssp245_total = aci_contribution_ssp245_total[:,:,0:aci_num]
aci_contribution_ssp585_total = aci_contribution_ssp585_total[:,:,0:aci_num]







aci_contribution_ssp245_county_hist = np.zeros((16,8,41,1000,aci_num))
aci_contribution_ssp585_county_hist = np.zeros((16,8,41,1000,aci_num))
aci_contribution_ssp245_county_future = np.zeros((16,8,79,1000,aci_num))
aci_contribution_ssp585_county_future = np.zeros((16,8,79,1000,aci_num))
for county in range(0,16):
    for model in range(0,8):
        aci_contribution_ssp245_county_hist[county,model,:,:,:] = np.split(locals()['aci_contribution_model_ssp245_copy_'+str(model)],16)[county][:,:,0:aci_num]+np.split(locals()['aci_contribution_model_ssp245_copy_'+str(model)],16)[county][:,:,aci_num:aci_num*2]
        aci_contribution_ssp585_county_hist[county,model,:,:,:] = np.split(locals()['aci_contribution_model_ssp585_copy_'+str(model)],16)[county][:,:,0:aci_num]+np.split(locals()['aci_contribution_model_ssp585_copy_'+str(model)],16)[county][:,:,aci_num:aci_num*2]
        aci_contribution_ssp245_county_future[county,model,:,:,:] = np.split(locals()['aci_contribution_model_ssp245_future_copy_'+str(model)],16)[county][:,:,0:aci_num]+np.split(locals()['aci_contribution_model_ssp245_future_copy_'+str(model)],16)[county][:,:,aci_num:aci_num*2]
        aci_contribution_ssp585_county_future[county,model,:,:,:] = np.split(locals()['aci_contribution_model_ssp585_future_copy_'+str(model)],16)[county][:,:,0:aci_num]+np.split(locals()['aci_contribution_model_ssp585_future_copy_'+str(model)],16)[county][:,:,aci_num:aci_num*2]
aci_contribution_ssp245_county = np.concatenate((aci_contribution_ssp245_county_hist, aci_contribution_ssp245_county_future),axis=2)
aci_contribution_ssp585_county = np.concatenate((aci_contribution_ssp585_county_hist, aci_contribution_ssp585_county_future),axis=2)

aci_contribution_ssp245_county_2050_change =  np.mean((np.mean(aci_contribution_ssp245_county[:,:,60:80,:,:], axis=2)-np.mean(aci_contribution_ssp245_county[:,:,21:41,:,:],axis=2)), axis=1)
aci_contribution_ssp245_county_2090_change =  np.mean((np.mean(aci_contribution_ssp245_county[:,:,-20:,:,:], axis=2)-np.mean(aci_contribution_ssp245_county[:,:,21:41,:,:],axis=2)), axis=1)
aci_contribution_ssp245_county_2050_change_percent = np.zeros((16,1000,aci_num))
aci_contribution_ssp245_county_2090_change_percent = np.zeros((16,1000,aci_num))

aci_contribution_ssp585_county_2050_change =  np.mean((np.mean(aci_contribution_ssp585_county[:,:,60:80,:,:], axis=2)-np.mean(aci_contribution_ssp585_county[:,:,21:41,:,:],axis=2)), axis=1)
aci_contribution_ssp585_county_2090_change =  np.mean((np.mean(aci_contribution_ssp585_county[:,:,-20:,:,:], axis=2)-np.mean(aci_contribution_ssp585_county[:,:,21:41,:,:],axis=2)), axis=1)
aci_contribution_ssp585_county_2050_change_percent = np.zeros((16,1000,aci_num))
aci_contribution_ssp585_county_2090_change_percent = np.zeros((16,1000,aci_num))

for county in range(0,16):
    for trial in range(0,1000):
        aci_contribution_ssp245_county_2050_change_percent[county,trial,:] = 100*aci_contribution_ssp245_county_2050_change[county,trial,:]/np.abs(np.sum(aci_contribution_ssp245_county_2050_change[county,trial,:]))
        aci_contribution_ssp245_county_2090_change_percent[county,trial,:] = 100*aci_contribution_ssp245_county_2090_change[county,trial,:]/np.abs(np.sum(aci_contribution_ssp245_county_2090_change[county,trial,:]))
        aci_contribution_ssp585_county_2050_change_percent[county,trial,:] = 100*aci_contribution_ssp585_county_2050_change[county,trial,:]/np.abs(np.sum(aci_contribution_ssp585_county_2050_change[county,trial,:]))
        aci_contribution_ssp585_county_2090_change_percent[county,trial,:] = 100*aci_contribution_ssp585_county_2090_change[county,trial,:]/np.abs(np.sum(aci_contribution_ssp585_county_2090_change[county,trial,:]))

np.save(str(save_path)+'aci_contribution_ssp245_total.npy',aci_contribution_ssp245_total)
np.save(str(save_path)+'aci_contribution_ssp585_total.npy',aci_contribution_ssp585_total)
np.save(str(save_path)+'aci_contribution_ssp245_county_total.npy',aci_contribution_ssp245_county)
np.save(str(save_path)+'aci_contribution_ssp585_county_total.npy',aci_contribution_ssp585_county)

np.save(str(save_path)+'aci_contribution_ssp245_county_2050_change_percent.npy',aci_contribution_ssp245_county_2050_change_percent)
np.save(str(save_path)+'aci_contribution_ssp585_county_2050_change_percent.npy',aci_contribution_ssp585_county_2050_change_percent)
np.save(str(save_path)+'aci_contribution_ssp245_county_2090_change_percent.npy',aci_contribution_ssp245_county_2090_change_percent)
np.save(str(save_path)+'aci_contribution_ssp585_county_2090_change_percent.npy',aci_contribution_ssp585_county_2090_change_percent)

np.save(str(save_path)+'aci_contribution_ssp245_county_2050_change.npy',aci_contribution_ssp245_county_2050_change)
np.save(str(save_path)+'aci_contribution_ssp585_county_2050_change.npy',aci_contribution_ssp585_county_2050_change)
np.save(str(save_path)+'aci_contribution_ssp245_county_2090_change.npy',aci_contribution_ssp245_county_2090_change)
np.save(str(save_path)+'aci_contribution_ssp585_county_2090_change.npy',aci_contribution_ssp585_county_2090_change)

