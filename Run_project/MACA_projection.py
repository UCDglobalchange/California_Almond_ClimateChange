##Code to use statistical relationship obtained from Lasso regression between gridMET-ACI and historical almond yield to 
##project almond yield based on MACA climate datasets for each county, and compute almond cropand area-weighted yield of California. 

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

home_path='/home/shqwu/California_Almond_ClimateChange-main/Run_project'
input_path_model = home_path+'/intermediate_data/lasso_model/'
input_path_MACA = home_path+'/intermediate_data/MACA_csv/'
input_path_area = home_path+'/input_data/'
save_path = home_path+'/output_data/projection/'
save_path_projection_csv = home_path+'/intermediate_data/projection_csv/'

aci_num = 14
for i in range(1,11):
    locals()['coef'+str(i)] = np.zeros((100,aci_num*2+32))
coef_sum = np.zeros((0,aci_num*2+32))
for i in range(1,1001):
    print(i)
    locals()['coef'+str(((i-1)//100)+1)][i%100-1] = genfromtxt(input_path_model+'coef_'+str(i)+'.csv', delimiter = ',')
for i in range(1,11):
    coef_sum = np.row_stack((coef_sum, locals()['coef'+str(i)]))


ACI_list = ['Dormancy_Freeze','Dormancy_ETo','Jan_Ppt','Bloom_Ppt','Bloom_Tmin','BloomFrostDays' ,'Bloom_ETo', 'Bloom_GDD4','Bloom_Humidity','Windy_days','Growing_ETo','GrowingGDD4', 'Growing_KDD30','FallTmean','FallTmean','harvest_Ppt']

county_list = ['Butte', 'Colusa', 'Fresno', 'Glenn', 'Kern', 'Kings', 'Madera', 'Merced', 'San Joaquin', 'Solano', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Yolo', 'Yuba']                      

model_list = ['bcc-csm1-1-m', 'bcc-csm1-1','BNU-ESM', 'CanESM2', 'CSIRO-Mk3-6-0', 'GFDL-ESM2G', 'GFDL-ESM2M', 'inmcm4', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR','CNRM-CM5', 'HadGEM2-CC365','HadGEM2-ES365', 'IPSL-CM5B-LR', 'MIROC5', 'MIROC-ESM', 'MIROC-ESM-CHEM','MRI-CGCM3']

area_csv = genfromtxt(input_path_area+'almond_area.csv', delimiter = ',')


##hist for new model
for trial in range(1,11):
    for model in range(0,18):
        simulation_rcp45 = np.zeros((656,100))
        simulation_rcp45_s = np.zeros((656,100))
        simulation_rcp45_m = np.zeros((656,100))
        simulation_rcp85 = np.zeros((656,100))
        simulation_rcp85_s = np.zeros((656,100))
        simulation_rcp85_m = np.zeros((656,100))
        aci_rcp45 = genfromtxt(input_path_MACA+str(model_list[model])+'hist_rcp45_ACI.csv', delimiter = ',')
        aci_rcp45_s = genfromtxt(input_path_MACA+str(model_list[model])+'hist_rcp45_s_ACI.csv', delimiter = ',')
        aci_rcp45_m = genfromtxt(input_path_MACA+str(model_list[model])+'hist_rcp45_m_ACI.csv', delimiter = ',')
        aci_rcp85 = genfromtxt(input_path_MACA+str(model_list[model])+'hist_rcp85_ACI.csv', delimiter = ',')
        aci_rcp85_s = genfromtxt(input_path_MACA+str(model_list[model])+'hist_rcp85_s_ACI.csv', delimiter = ',')
        aci_rcp85_m = genfromtxt(input_path_MACA+str(model_list[model])+'hist_rcp85_m_ACI.csv', delimiter = ',')
        for i in range(0,656):
            for j in range(0,100):
                simulation_rcp45[i,j] = np.nansum(aci_rcp45[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_rcp45_s[i,j] = np.nansum(aci_rcp45_s[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_rcp45_m[i,j] = np.nansum(aci_rcp45_m[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_rcp85[i,j] = np.nansum(aci_rcp85[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_rcp85_s[i,j] = np.nansum(aci_rcp85_s[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_rcp85_m[i,j] = np.nansum(aci_rcp85_m[i,:]*locals()['coef'+str(trial)][j,:])
        savetxt(save_path_projection_csv+str(model_list[model])+'_hist_prediction_'+str(trial)+'_rcp45.csv', simulation_rcp45, delimiter = ',')
        savetxt(save_path_projection_csv+str(model_list[model])+'_hist_prediction_'+str(trial)+'_rcp45_s.csv', simulation_rcp45_s, delimiter = ',')
        savetxt(save_path_projection_csv+str(model_list[model])+'_hist_prediction_'+str(trial)+'_rcp45_m.csv', simulation_rcp45_m, delimiter = ',')
        savetxt(save_path_projection_csv+str(model_list[model])+'_hist_prediction_'+str(trial)+'_rcp85.csv', simulation_rcp85, delimiter = ',')
        savetxt(save_path_projection_csv+str(model_list[model])+'_hist_prediction_'+str(trial)+'_rcp85_s.csv', simulation_rcp85_s, delimiter = ',')
        savetxt(save_path_projection_csv+str(model_list[model])+'_hist_prediction_'+str(trial)+'_rcp85_m.csv', simulation_rcp85_m, delimiter = ',')

##future
for trial in range(1,11):
    for model in range(0,18):
        simulation_rcp45 = np.zeros((1264,100))
        simulation_rcp45_s = np.zeros((1264,100))
        simulation_rcp45_m = np.zeros((1264,100))
        simulation_rcp85 = np.zeros((1264,100))
        simulation_rcp85_s = np.zeros((1264,100))
        simulation_rcp85_m = np.zeros((1264,100))
        aci_rcp45 = genfromtxt(input_path_MACA+str(model_list[model])+'future_rcp45_ACI.csv', delimiter = ',')
        aci_rcp45_s = genfromtxt(input_path_MACA+str(model_list[model])+'future_rcp45_s_ACI.csv', delimiter = ',')
        aci_rcp45_m = genfromtxt(input_path_MACA+str(model_list[model])+'future_rcp45_m_ACI.csv', delimiter = ',')
        aci_rcp85 = genfromtxt(input_path_MACA+str(model_list[model])+'future_rcp85_ACI.csv', delimiter = ',')
        aci_rcp85_s = genfromtxt(input_path_MACA+str(model_list[model])+'future_rcp85_s_ACI.csv', delimiter = ',')
        aci_rcp85_m = genfromtxt(input_path_MACA+str(model_list[model])+'future_rcp85_m_ACI.csv', delimiter = ',')
        for i in range(0,1264):
            for j in range(0,100):
                simulation_rcp45[i,j] = np.nansum(aci_rcp45[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_rcp45_s[i,j] = np.nansum(aci_rcp45_s[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_rcp45_m[i,j] = np.nansum(aci_rcp45_m[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_rcp85[i,j] = np.nansum(aci_rcp85[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_rcp85_s[i,j] = np.nansum(aci_rcp85_s[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_rcp85_m[i,j] = np.nansum(aci_rcp85_m[i,:]*locals()['coef'+str(trial)][j,:])
        savetxt(save_path_projection_csv+str(model_list[model])+'_future_prediction_'+str(trial)+'_rcp45.csv', simulation_rcp45, delimiter = ',')
        savetxt(save_path_projection_csv+str(model_list[model])+'_future_prediction_'+str(trial)+'_rcp45_s.csv', simulation_rcp45_s, delimiter = ',')
        savetxt(save_path_projection_csv+str(model_list[model])+'_future_prediction_'+str(trial)+'_rcp45_m.csv', simulation_rcp45_m, delimiter = ',')
        savetxt(save_path_projection_csv+str(model_list[model])+'_future_prediction_'+str(trial)+'_rcp85.csv', simulation_rcp85, delimiter = ',')
        savetxt(save_path_projection_csv+str(model_list[model])+'_future_prediction_'+str(trial)+'_rcp85_s.csv', simulation_rcp85_s, delimiter = ',')
        savetxt(save_path_projection_csv+str(model_list[model])+'_future_prediction_'+str(trial)+'_rcp85_m.csv', simulation_rcp85_m, delimiter = ',')




area = area_csv[0:41,:]

yield_all_hist = np.zeros((656,0))
##hist rcp45
average_model = np.zeros((656,1000))
for model_id in range(0,18):
    locals()['model_'+str(model_id)] = np.zeros((656,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_hist'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_hist_prediction_'+str(trial)+'_rcp45.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_hist'][locals()[str(model_list[model_id])+str(trial)+'_hist']<0] = 0
        yield_all_hist = np.column_stack((yield_all_hist,locals()[str(model_list[model_id])+str(trial)+'_hist']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_hist']))
    locals()['model_'+str(model_id)+'_average'] = np.nanmedian(locals()['model_'+str(model_id)], axis = 1) ## change to median 
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/18
average_model_hist = average_model
production_average_model = np.zeros((656,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average,model_7_average,model_8_average, model_9_average, model_10_average, model_11_average, model_12_average, model_13_average, model_14_average, model_15_average, model_16_average, model_17_average))
yield_all_model_hist_rcp45 = yield_all_hist
yield_all_model_hist_rcp45_average_model = yield_all_model
production_all_model = np.zeros((656,18))
production_all_hist = np.zeros((656,18000))
yield_1980_rcp45 = np.zeros((16))
for index in range(0,16):
    yield_1980_rcp45[index] = np.mean(yield_all_model[index * 41,:])
for index in range(0,16):
    for year in range(0,41):
        production_all_model[index*41+year,:] = yield_all_model[index*41+year,:]*area[year,index]
        production_average_model[index*41+year,:] = average_model[index*41+year,:]*area[year,index]
        production_all_hist[index*41+year,:] = yield_all_hist[index*41+year,:]*area[year,index]
production_all_hist_split = np.split(production_all_hist, 16)
production_model_split = np.split(production_all_model,16) 
production_model_split_hist = production_model_split
production_across_state_hist = np.zeros((41,18))
production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_hist = np.zeros((41,1000))
production_all_hist_split_across_state = np.zeros((41,18000)) 
for county_id in range(0,16):
    for year in range(0,41):
        production_across_state_hist[year,:] = production_across_state_hist[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_hist[year,:] = production_average_model_across_state_hist[year,:]+production_average_model_split[county_id][year,:]
        production_all_hist_split_across_state[year, :] = production_all_hist_split_across_state[year,:]+production_all_hist_split[county_id][year,:]
production_all_rcp45_hist = production_all_hist_split_across_state
yield_across_state_hist_rcp45 =np.zeros((41,18))
yield_average_model_hist_rcp45 = np.zeros((41,1000))
yield_all_hist_rcp45 = np.zeros((41,18000))
for year in range(0,41):
    yield_across_state_hist_rcp45[year,:] = production_across_state_hist[year,:]/np.sum(area[year])
    yield_average_model_hist_rcp45[year,:] = production_average_model_across_state_hist[year,:]/np.sum(area[year])
    yield_all_hist_rcp45[year,:] = production_all_hist_split_across_state[year,:]/np.sum(area[year])
np.save(save_path+'yield_across_state_hist_rcp45.npy',yield_across_state_hist_rcp45)
np.save(save_path+'yield_average_model_hist_rcp45.npy',yield_average_model_hist_rcp45)
np.save(save_path+'yield_all_hist_rcp45.npy',yield_all_hist_rcp45)
np.save(save_path+'yield_1980_rcp45.npy',yield_1980_rcp45)



yield_all_hist = np.zeros((656,0))
##hist rcp45
average_model = np.zeros((656,1000))
for model_id in range(0,18):
    locals()['model_'+str(model_id)] = np.zeros((656,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_hist'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_hist_prediction_'+str(trial)+'_rcp45_s.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_hist'][locals()[str(model_list[model_id])+str(trial)+'_hist']<0] = 0
        yield_all_hist = np.column_stack((yield_all_hist,locals()[str(model_list[model_id])+str(trial)+'_hist']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_hist']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1) ## change to median 
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/18
average_model_hist = average_model
production_average_model = np.zeros((656,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average,model_7_average,model_8_average, model_9_average, model_10_average, model_11_average, model_12_average, model_13_average, model_14_average, model_15_average, model_16_average, model_17_average))
yield_all_model_hist_rcp45_s = yield_all_hist
yield_all_model_hist_rcp45_s_average_model = yield_all_model
production_all_model = np.zeros((656,18))
production_all_hist = np.zeros((656,18000))
for index in range(0,16):
    for year in range(0,41):
        production_all_model[index*41+year,:] = yield_all_model[index*41+year,:]*area[year,index]
        production_average_model[index*41+year,:] = average_model[index*41+year,:]*area[year,index]
        production_all_hist[index*41+year,:] = yield_all_hist[index*41+year,:]*area[year,index]

production_all_hist_split = np.split(production_all_hist, 16)
production_model_split = np.split(production_all_model,16) 
production_model_split_hist = production_model_split
production_across_state_hist = np.zeros((41,18))
production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_hist = np.zeros((41,1000))
production_all_hist_split_across_state = np.zeros((41,18000))
for county_id in range(0,16):
    for year in range(0,41):
        production_across_state_hist[year,:] = production_across_state_hist[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_hist[year,:] = production_average_model_across_state_hist[year,:]+production_average_model_split[county_id][year,:]
        production_all_hist_split_across_state[year, :] = production_all_hist_split_across_state[year,:]+production_all_hist_split[county_id][year,:]
production_all_rcp45_s_hist = production_all_hist_split_across_state
yield_across_state_hist_rcp45_s =np.zeros((41,18))
yield_average_model_hist_rcp45_s = np.zeros((41,1000))
yield_all_hist_rcp45_s = np.zeros((41,18000))
for year in range(0,41):
    yield_across_state_hist_rcp45_s[year,:] = production_across_state_hist[year,:]/np.sum(area[year])
    yield_average_model_hist_rcp45_s[year,:] = production_average_model_across_state_hist[year,:]/np.sum(area[year])
    yield_all_hist_rcp45_s[year,:] = production_all_hist_split_across_state[year,:]/np.sum(area[year])
np.save(save_path+'yield_across_state_hist_rcp45_s.npy',yield_across_state_hist_rcp45_s)
np.save(save_path+'yield_average_model_hist_rcp45_s.npy',yield_average_model_hist_rcp45_s)
np.save(save_path+'yield_all_hist_rcp45_s.npy',yield_all_hist_rcp45_s)


yield_all_hist = np.zeros((656,0))
##hist rcp45_m
average_model = np.zeros((656,1000))
for model_id in range(0,18):
    locals()['model_'+str(model_id)] = np.zeros((656,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_hist'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_hist_prediction_'+str(trial)+'_rcp45_m.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_hist'][locals()[str(model_list[model_id])+str(trial)+'_hist']<0] = 0
        yield_all_hist = np.column_stack((yield_all_hist,locals()[str(model_list[model_id])+str(trial)+'_hist']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_hist']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1) ## change to median
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/18
average_model_hist = average_model
production_average_model = np.zeros((656,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average,model_7_average,model_8_average, model_9_average, model_10_average, model_11_average, model_12_average, model_13_average, model_14_average, model_15_average, model_16_average, model_17_average))
yield_all_model_hist_rcp45_m = yield_all_hist
yield_all_model_hist_rcp45_m_average_model = yield_all_model
production_all_model = np.zeros((656,18))
production_all_hist = np.zeros((656,18000))
for index in range(0,16):
    for year in range(0,41):
        production_all_model[index*41+year,:] = yield_all_model[index*41+year,:]*area[year,index]
        production_average_model[index*41+year,:] = average_model[index*41+year,:]*area[year,index]
        production_all_hist[index*41+year,:] = yield_all_hist[index*41+year,:]*area[year,index]

production_all_hist_split = np.split(production_all_hist, 16)
production_model_split = np.split(production_all_model,16)
production_model_split_hist = production_model_split
production_across_state_hist = np.zeros((41,18))
production_average_model_split = np.split(production_average_model,16)
production_average_model_across_state_hist = np.zeros((41,1000))
production_all_hist_split_across_state = np.zeros((41,18000))
for county_id in range(0,16):
    for year in range(0,41):
        production_across_state_hist[year,:] = production_across_state_hist[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_hist[year,:] = production_average_model_across_state_hist[year,:]+production_average_model_split[county_id][year,:]
        production_all_hist_split_across_state[year, :] = production_all_hist_split_across_state[year,:]+production_all_hist_split[county_id][year,:]
production_all_rcp45_m_hist = production_all_hist_split_across_state
yield_across_state_hist_rcp45_m =np.zeros((41,18))
yield_average_model_hist_rcp45_m = np.zeros((41,1000))
yield_all_hist_rcp45_m = np.zeros((41,18000))
for year in range(0,41):
    yield_across_state_hist_rcp45_m[year,:] = production_across_state_hist[year,:]/np.sum(area[year])
    yield_average_model_hist_rcp45_m[year,:] = production_average_model_across_state_hist[year,:]/np.sum(area[year])
    yield_all_hist_rcp45_m[year,:] = production_all_hist_split_across_state[year,:]/np.sum(area[year])
np.save(save_path+'yield_across_state_hist_rcp45_m.npy',yield_across_state_hist_rcp45_m)
np.save(save_path+'yield_average_model_hist_rcp45_m.npy',yield_average_model_hist_rcp45_m)
np.save(save_path+'yield_all_hist_rcp45_m.npy',yield_all_hist_rcp45_m)



##hist rcp85
yield_all_hist = np.zeros((656,0))
average_model = np.zeros((656,1000))
for model_id in range(0,18):
    locals()['model_'+str(model_id)] = np.zeros((656,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_hist'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_hist_prediction_'+str(trial)+'_rcp85.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_hist'][locals()[str(model_list[model_id])+str(trial)+'_hist']<0] = 0
        yield_all_hist = np.column_stack((yield_all_hist,locals()[str(model_list[model_id])+str(trial)+'_hist']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_hist']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1) ## change to median 
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/18
average_model_hist = average_model
production_average_model = np.zeros((656,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average,model_7_average,model_8_average, model_9_average, model_10_average, model_11_average, model_12_average, model_13_average, model_14_average, model_15_average, model_16_average, model_17_average))
yield_all_model_hist_rcp85 = yield_all_hist
yield_all_model_hist_rcp85_average_model = yield_all_model
production_all_model = np.zeros((656,18))
production_all_hist = np.zeros((656,18000))
yield_1980_rcp85 = np.zeros((16))
for index in range(0,16):
    yield_1980_rcp85[index] = np.mean(yield_all_model[index * 41,:])
for index in range(0,16):
    for year in range(0,41):
        production_all_model[index*41+year,:] = yield_all_model[index*41+year,:]*area[year,index%16]
        production_average_model[index*41+year,:] = average_model[index*41+year,:]*area[year,index%16]
        production_all_hist[index*41+year,:] = yield_all_hist[index*41+year,:]*area[year,index%16]
production_all_hist_split = np.split(production_all_hist, 16)
production_model_split = np.split(production_all_model,16) 
production_model_split_hist = production_model_split
production_across_state_hist = np.zeros((41,18))
production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_hist = np.zeros((41,1000))
production_all_hist_split_across_state = np.zeros((41,18000))
for county_id in range(0,16):
    for year in range(0,41):
        production_across_state_hist[year,:] = production_across_state_hist[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_hist[year,:] = production_average_model_across_state_hist[year,:]+production_average_model_split[county_id][year,:]
        production_all_hist_split_across_state[year, :] = production_all_hist_split_across_state[year,:]+production_all_hist_split[county_id][year,:]
production_all_rcp85_hist = production_all_hist_split_across_state
yield_across_state_hist_rcp85 =np.zeros((41,18))
yield_average_model_hist_rcp85 = np.zeros((41,1000))
yield_all_hist_rcp85 = np.zeros((41,18000))
for year in range(0,41):
    yield_across_state_hist_rcp85[year,:] = production_across_state_hist[year,:]/np.sum(area[year])
    yield_average_model_hist_rcp85[year,:] = production_average_model_across_state_hist[year,:]/np.sum(area[year])
    yield_all_hist_rcp85[year,:] = production_all_hist_split_across_state[year,:]/np.sum(area[year])
np.save(save_path+'yield_across_state_hist_rcp85.npy',yield_across_state_hist_rcp85)
np.save(save_path+'yield_average_model_hist_rcp85.npy',yield_average_model_hist_rcp85)
np.save(save_path+'yield_all_hist_rcp85.npy',yield_all_hist_rcp85)
np.save(save_path+'yield_1980_rcp85.npy',yield_1980_rcp85)





yield_all_hist = np.zeros((656,0))
##hist rcp85_s
average_model = np.zeros((656,1000))
for model_id in range(0,18):
    locals()['model_'+str(model_id)] = np.zeros((656,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_hist'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_hist_prediction_'+str(trial)+'_rcp85_s.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_hist'][locals()[str(model_list[model_id])+str(trial)+'_hist']<0] = 0
        yield_all_hist = np.column_stack((yield_all_hist,locals()[str(model_list[model_id])+str(trial)+'_hist']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_hist']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1) ## change to median 
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/18
average_model_hist = average_model
production_average_model = np.zeros((656,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average,model_7_average,model_8_average, model_9_average, model_10_average, model_11_average, model_12_average, model_13_average, model_14_average, model_15_average, model_16_average, model_17_average))
yield_all_model_hist_rcp85_s = yield_all_hist
yield_all_model_hist_rcp85_s_average_model = yield_all_model
production_all_model = np.zeros((656,18))
production_all_hist = np.zeros((656,18000))
for index in range(0,16):
    for year in range(0,41):
        production_all_model[index*41+year,:] = yield_all_model[index*41+year,:]*area[year,index]
        production_average_model[index*41+year,:] = average_model[index*41+year,:]*area[year,index]
        production_all_hist[index*41+year,:] = yield_all_hist[index*41+year,:]*area[year,index]

production_all_hist_split = np.split(production_all_hist, 16)
production_model_split = np.split(production_all_model,16) 
production_model_split_hist = production_model_split
production_across_state_hist = np.zeros((41,18))
production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_hist = np.zeros((41,1000))
production_all_hist_split_across_state = np.zeros((41,18000))
for county_id in range(0,16):
    for year in range(0,41):
        production_across_state_hist[year,:] = production_across_state_hist[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_hist[year,:] = production_average_model_across_state_hist[year,:]+production_average_model_split[county_id][year,:]
        production_all_hist_split_across_state[year, :] = production_all_hist_split_across_state[year,:]+production_all_hist_split[county_id][year,:]
production_all_rcp85_s_hist = production_all_hist_split_across_state
yield_across_state_hist_rcp85_s =np.zeros((41,18))
yield_average_model_hist_rcp85_s = np.zeros((41,1000))
yield_all_hist_rcp85_s = np.zeros((41,18000))
for year in range(0,41):
    yield_across_state_hist_rcp85_s[year,:] = production_across_state_hist[year,:]/np.sum(area[year])
    yield_average_model_hist_rcp85_s[year,:] = production_average_model_across_state_hist[year,:]/np.sum(area[year])
    yield_all_hist_rcp85_s[year,:] = production_all_hist_split_across_state[year,:]/np.sum(area[year])
np.save(save_path+'yield_across_state_hist_rcp85_s.npy',yield_across_state_hist_rcp85_s)
np.save(save_path+'yield_average_model_hist_rcp85_s.npy',yield_average_model_hist_rcp85_s)
np.save(save_path+'yield_all_hist_rcp85_s.npy',yield_all_hist_rcp85_s)


##hist rcp85 m
average_model = np.zeros((656,1000))
yield_all_hist = np.zeros((656,0))
for model_id in range(0,18):
    locals()['model_'+str(model_id)] = np.zeros((656,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_hist'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_hist_prediction_'+str(trial)+'_rcp85_m.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_hist'][locals()[str(model_list[model_id])+str(trial)+'_hist']<0] = 0
        yield_all_hist = np.column_stack((yield_all_hist,locals()[str(model_list[model_id])+str(trial)+'_hist']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_hist']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1) ## change to median
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/18
average_model_hist = average_model
production_average_model = np.zeros((656,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average,model_7_average,model_8_average, model_9_average, model_10_average, model_11_average, model_12_average, model_13_average, model_14_average, model_15_average, model_16_average, model_17_average))
yield_all_model_hist_rcp85_m = yield_all_hist
yield_all_model_hist_rcp85_m_average_model = yield_all_model
production_all_model = np.zeros((656,18))
production_all_hist = np.zeros((656,18000))
for index in range(0,16):
    for year in range(0,41):
        production_all_model[index*41+year,:] = yield_all_model[index*41+year,:]*area[year,index]
        production_average_model[index*41+year,:] = average_model[index*41+year,:]*area[year,index]
        production_all_hist[index*41+year,:] = yield_all_hist[index*41+year,:]*area[year,index]

production_all_hist_split = np.split(production_all_hist, 16)
production_model_split = np.split(production_all_model,16)
production_model_split_hist = production_model_split
production_across_state_hist = np.zeros((41,18))
production_average_model_split = np.split(production_average_model,16)
production_average_model_across_state_hist = np.zeros((41,1000))
production_all_hist_split_across_state = np.zeros((41,18000))
for county_id in range(0,16):
    for year in range(0,41):
        production_across_state_hist[year,:] = production_across_state_hist[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_hist[year,:] = production_average_model_across_state_hist[year,:]+production_average_model_split[county_id][year,:]
        production_all_hist_split_across_state[year, :] = production_all_hist_split_across_state[year,:]+production_all_hist_split[county_id][year,:]
production_all_rcp85_m_hist = production_all_hist_split_across_state
yield_across_state_hist_rcp85_m =np.zeros((41,18))
yield_average_model_hist_rcp85_m = np.zeros((41,1000))
yield_all_hist_rcp85_m = np.zeros((41,18000))
for year in range(0,41):
    yield_across_state_hist_rcp85_m[year,:] = production_across_state_hist[year,:]/np.sum(area[year])
    yield_average_model_hist_rcp85_m[year,:] = production_average_model_across_state_hist[year,:]/np.sum(area[year])
    yield_all_hist_rcp85_m[year,:] = production_all_hist_split_across_state[year,:]/np.sum(area[year])
np.save(save_path+'yield_across_state_hist_rcp85_m.npy',yield_across_state_hist_rcp85_m)
np.save(save_path+'yield_average_model_hist_rcp85_m.npy',yield_average_model_hist_rcp85_m)
np.save(save_path+'yield_all_hist_rcp85_m.npy',yield_all_hist_rcp85_m)




area = area[-1]
#future rcp45
yield_all = np.zeros((1264,0))
average_model = np.zeros((1264,1000))
for model_id in range(0,18):
    locals()['model_'+str(model_id)] = np.zeros((1264,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_rcp45'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_future_prediction_'+str(trial)+'_rcp45.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_rcp45'][locals()[str(model_list[model_id])+str(trial)+'_rcp45']<0] = 0
        yield_all = np.column_stack((yield_all,locals()[str(model_list[model_id])+str(trial)+'_rcp45']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_rcp45']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1)
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/18
average_model_rcp45 = average_model
production_average_model = np.zeros((1264,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average,model_7_average,model_8_average, model_9_average, model_10_average, model_11_average, model_12_average, model_13_average, model_14_average, model_15_average, model_16_average, model_17_average))
yield_all_model_future_rcp45 = yield_all
production_all_model = np.zeros((1264,18))
production_all = np.zeros((1264,18000))
for index in range(0,16):
    for year in range(0,79):
        production_all_model[index*79+year,:] = yield_all_model[index*79+year,:]*area[index]
        production_average_model[index*79+year,:] = average_model[index*79+year,:]*area[index]
        production_all[index*79+year,:] = yield_all[index*79+year,:]*area[index]
production_all_split = np.split(production_all, 16)
production_model_split = np.split(production_all_model,16) 
production_model_split_45 = production_model_split
production_across_state_rcp45 = np.zeros((79,18))
production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_rcp45 = np.zeros((79,1000))
production_all_split_across_state = np.zeros((79,18000))
for county_id in range(0,16):
    for year in range(0,79):
        production_across_state_rcp45[year,:] = production_across_state_rcp45[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_rcp45[year,:] = production_average_model_across_state_rcp45[year,:]+production_average_model_split[county_id][year,:]
        production_all_split_across_state[year, :] = production_all_split_across_state[year,:]+production_all_split[county_id][year,:]
production_all_rcp45_future = production_all_split_across_state
yield_across_state_future_rcp45 =np.zeros((79,18))
yield_average_model_future_rcp45 = np.zeros((79,1000))
yield_all_future_rcp45 = np.zeros((79,18000))

for year in range(0,79):
    yield_across_state_future_rcp45[year,:] = production_across_state_rcp45[year,:]/np.sum(area)
    yield_average_model_future_rcp45[year,:] = production_average_model_across_state_rcp45[year,:]/np.sum(area)
    yield_all_future_rcp45[year,:] = production_all_split_across_state[year,:]/np.sum(area)
np.save(save_path+'yield_across_state_future_rcp45.npy',yield_across_state_future_rcp45)
np.save(save_path+'yield_average_model_future_rcp45.npy',yield_average_model_future_rcp45)
np.save(save_path+'yield_all_future_rcp45.npy',yield_all_future_rcp45)


# future rcp45_s
yield_all = np.zeros((1264,0))
average_model = np.zeros((1264,1000))
for model_id in range(0,18):
    print(model_id)
    locals()['model_'+str(model_id)] = np.zeros((1264,0))
    for trial in range(1,11):
        print(trial)
        locals()[str(model_list[model_id])+str(trial)+'_rcp45'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_future_prediction_'+str(trial)+'_rcp45_s.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_rcp45'][locals()[str(model_list[model_id])+str(trial)+'_rcp45']<0] = 0
        yield_all = np.column_stack((yield_all,locals()[str(model_list[model_id])+str(trial)+'_rcp45']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_rcp45']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1)
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/18
average_model_rcp45 = average_model
production_average_model = np.zeros((1264,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average,model_7_average,model_8_average, model_9_average, model_10_average, model_11_average, model_12_average, model_13_average, model_14_average, model_15_average, model_16_average, model_17_average))
yield_all_model_future_rcp45_s = yield_all
production_all_model = np.zeros((1264,18))
production_all = np.zeros((1264,18000))
for index in range(0,16):
    for year in range(0,79):
        production_all_model[index*79+year,:] = yield_all_model[index*79+year,:]*area[index]
        production_average_model[index*79+year,:] = average_model[index*79+year,:]*area[index]
        production_all[index*79+year,:] = yield_all[index*79+year,:]*area[index]
production_all_split = np.split(production_all, 16)
production_model_split = np.split(production_all_model,16) 
production_model_split_45 = production_model_split
production_across_state_rcp45 = np.zeros((79,18))
production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_rcp45 = np.zeros((79,1000))
production_all_split_across_state = np.zeros((79,18000))
for county_id in range(0,16):
    for year in range(0,79):
        production_across_state_rcp45[year,:] = production_across_state_rcp45[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_rcp45[year,:] = production_average_model_across_state_rcp45[year,:]+production_average_model_split[county_id][year,:]
        production_all_split_across_state[year, :] = production_all_split_across_state[year,:]+production_all_split[county_id][year,:]
production_all_rcp45_s_future = production_all_split_across_state
yield_across_state_future_rcp45_s =np.zeros((79,18))
yield_average_model_future_rcp45_s = np.zeros((79,1000))
yield_all_future_rcp45_s = np.zeros((79,18000))

for year in range(0,79):
    yield_across_state_future_rcp45_s[year,:] = production_across_state_rcp45[year,:]/np.sum(area)
    yield_average_model_future_rcp45_s[year,:] = production_average_model_across_state_rcp45[year,:]/np.sum(area)
    yield_all_future_rcp45_s[year,:] = production_all_split_across_state[year,:]/np.sum(area)
np.save(save_path+'yield_across_state_future_rcp45_s.npy',yield_across_state_future_rcp45_s)
np.save(save_path+'yield_average_model_future_rcp45_s.npy',yield_average_model_future_rcp45_s)
np.save(save_path+'yield_all_future_rcp45_s.npy',yield_all_future_rcp45_s)

# future rcp45_m
yield_all = np.zeros((1264,0))
average_model = np.zeros((1264,1000))
for model_id in range(0,18):
    print(model_id)
    locals()['model_'+str(model_id)] = np.zeros((1264,0))
    for trial in range(1,11):
        print(trial)
        locals()[str(model_list[model_id])+str(trial)+'_rcp45'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_future_prediction_'+str(trial)+'_rcp45_m.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_rcp45'][locals()[str(model_list[model_id])+str(trial)+'_rcp45']<0] = 0
        yield_all = np.column_stack((yield_all,locals()[str(model_list[model_id])+str(trial)+'_rcp45']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_rcp45']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1)
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/18
average_model_rcp45 = average_model
production_average_model = np.zeros((1264,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average,model_7_average,model_8_average, model_9_average, model_10_average, model_11_average, model_12_average, model_13_average, model_14_average, model_15_average, model_16_average, model_17_average))
yield_all_model_future_rcp45_m = yield_all
production_all_model = np.zeros((1264,18))
production_all = np.zeros((1264,18000))
for index in range(0,16):
    for year in range(0,79):
        production_all_model[index*79+year,:] = yield_all_model[index*79+year,:]*area[index]
        production_average_model[index*79+year,:] = average_model[index*79+year,:]*area[index]
        production_all[index*79+year,:] = yield_all[index*79+year,:]*area[index]
production_all_split = np.split(production_all, 16)
production_model_split = np.split(production_all_model,16)
production_model_split_45 = production_model_split
production_across_state_rcp45 = np.zeros((79,18))
production_average_model_split = np.split(production_average_model,16)
production_average_model_across_state_rcp45 = np.zeros((79,1000))
production_all_split_across_state = np.zeros((79,18000))
for county_id in range(0,16):
    for year in range(0,79):
        production_across_state_rcp45[year,:] = production_across_state_rcp45[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_rcp45[year,:] = production_average_model_across_state_rcp45[year,:]+production_average_model_split[county_id][year,:]
        production_all_split_across_state[year, :] = production_all_split_across_state[year,:]+production_all_split[county_id][year,:]
production_all_rcp45_m_future = production_all_split_across_state
yield_across_state_future_rcp45_m =np.zeros((79,18))
yield_average_model_future_rcp45_m = np.zeros((79,1000))
yield_all_future_rcp45_m = np.zeros((79,18000))

for year in range(0,79):
    yield_across_state_future_rcp45_m[year,:] = production_across_state_rcp45[year,:]/np.sum(area)
    yield_average_model_future_rcp45_m[year,:] = production_average_model_across_state_rcp45[year,:]/np.sum(area)
    yield_all_future_rcp45_m[year,:] = production_all_split_across_state[year,:]/np.sum(area)
np.save(save_path+'yield_across_state_future_rcp45_m.npy',yield_across_state_future_rcp45_m)
np.save(save_path+'yield_average_model_future_rcp45_m.npy',yield_average_model_future_rcp45_m)
np.save(save_path+'yield_all_future_rcp45_m.npy',yield_all_future_rcp45_m)



#rcp85
yield_all = np.zeros((1264,0))
average_model = np.zeros((1264,1000))
for model_id in range(0,18):
    locals()['model_'+str(model_id)] = np.zeros((1264,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_rcp85'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_future_prediction_'+str(trial)+'_rcp85.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_rcp85'][locals()[str(model_list[model_id])+str(trial)+'_rcp85']<0] = 0
        yield_all = np.column_stack((yield_all,locals()[str(model_list[model_id])+str(trial)+'_rcp85']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_rcp85']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1)
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/18
average_model_rcp85 = average_model
production_average_model = np.zeros((1264,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average,model_7_average,model_8_average, model_9_average, model_10_average, model_11_average, model_12_average, model_13_average, model_14_average, model_15_average, model_16_average, model_17_average))
yield_all_model_future_rcp85 = yield_all
production_all_model = np.zeros((1264,18))
production_all = np.zeros((1264,18000))
for index in range(0,16):
    for year in range(0,79):
        production_all_model[index*79+year,:] = yield_all_model[index*79+year,:]*area[index]
        production_average_model[index*79+year,:] = average_model[index*79+year,:]*area[index]
        production_all[index*79+year,:] = yield_all[index*79+year,:]*area[index]
production_all_split = np.split(production_all, 16)
production_model_split = np.split(production_all_model,16) 
production_model_split_85 = production_model_split
production_across_state_rcp85 = np.zeros((79,18))
production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_rcp85 = np.zeros((79,1000))
production_all_split_across_state = np.zeros((79,18000))
for county_id in range(0,16):
    for year in range(0,79):
        production_across_state_rcp85[year,:] = production_across_state_rcp85[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_rcp85[year,:] = production_average_model_across_state_rcp85[year,:]+production_average_model_split[county_id][year,:]
        production_all_split_across_state[year, :] = production_all_split_across_state[year,:]+production_all_split[county_id][year,:]
production_all_rcp85_future = production_all_split_across_state
yield_across_state_future_rcp85 =np.zeros((79,18))
yield_average_model_future_rcp85 = np.zeros((79,1000))
yield_all_future_rcp85 = np.zeros((79,18000))

for year in range(0,79):
    yield_across_state_future_rcp85[year,:] = production_across_state_rcp85[year,:]/np.sum(area)
    yield_average_model_future_rcp85[year,:] = production_average_model_across_state_rcp85[year,:]/np.sum(area)
    yield_all_future_rcp85[year,:] = production_all_split_across_state[year,:]/np.sum(area)
np.save(save_path+'yield_across_state_future_rcp85.npy',yield_across_state_future_rcp85)
np.save(save_path+'yield_average_model_future_rcp85.npy',yield_average_model_future_rcp85)
np.save(save_path+'yield_all_future_rcp85.npy',yield_all_future_rcp85)

#rcp85_stop_tech
yield_all = np.zeros((1264,0))
average_model = np.zeros((1264,1000))
for model_id in range(0,18):
    locals()['model_'+str(model_id)] = np.zeros((1264,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_rcp85'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_future_prediction_'+str(trial)+'_rcp85_s.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_rcp85'][locals()[str(model_list[model_id])+str(trial)+'_rcp85']<0] = 0
        yield_all = np.column_stack((yield_all,locals()[str(model_list[model_id])+str(trial)+'_rcp85']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_rcp85']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1)
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/18
average_model_rcp85 = average_model
production_average_model = np.zeros((1264,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average,model_7_average,model_8_average, model_9_average, model_10_average, model_11_average, model_12_average, model_13_average, model_14_average, model_15_average, model_16_average, model_17_average))
yield_all_model_future_rcp85_s = yield_all
production_all_model = np.zeros((1264,18))
production_all = np.zeros((1264,18000))
for index in range(0,16):
    for year in range(0,79):
        production_all_model[index*79+year,:] = yield_all_model[index*79+year,:]*area[index]
        production_average_model[index*79+year,:] = average_model[index*79+year,:]*area[index]
        production_all[index*79+year,:] = yield_all[index*79+year,:]*area[index]
production_all_split = np.split(production_all, 16)
production_model_split = np.split(production_all_model,16) 
production_model_split_85 = production_model_split
production_across_state_rcp85 = np.zeros((79,18))
production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_rcp85 = np.zeros((79,1000))
production_all_split_across_state = np.zeros((79,18000))
for county_id in range(0,16):
    for year in range(0,79):
        production_across_state_rcp85[year,:] = production_across_state_rcp85[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_rcp85[year,:] = production_average_model_across_state_rcp85[year,:]+production_average_model_split[county_id][year,:]
        production_all_split_across_state[year, :] = production_all_split_across_state[year,:]+production_all_split[county_id][year,:]
production_all_rcp85_s_future = production_all_split_across_state
yield_across_state_future_rcp85_s =np.zeros((79,18))
yield_average_model_future_rcp85_s = np.zeros((79,1000))
yield_all_future_rcp85_s = np.zeros((79,18000))

for year in range(0,79):
    yield_across_state_future_rcp85_s[year,:] = production_across_state_rcp85[year,:]/np.sum(area)
    yield_average_model_future_rcp85_s[year,:] = production_average_model_across_state_rcp85[year,:]/np.sum(area)
    yield_all_future_rcp85_s[year,:] = production_all_split_across_state[year,:]/np.sum(area)

np.save(save_path+'yield_across_state_future_rcp85_s.npy',yield_across_state_future_rcp85_s)
np.save(save_path+'yield_average_model_future_rcp85_s.npy',yield_average_model_future_rcp85_s)
np.save(save_path+'yield_all_future_rcp85_s.npy',yield_all_future_rcp85_s)



#rcp85_m
yield_all = np.zeros((1264,0))
average_model = np.zeros((1264,1000))
for model_id in range(0,18):
    locals()['model_'+str(model_id)] = np.zeros((1264,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_rcp85'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_future_prediction_'+str(trial)+'_rcp85_m.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_rcp85'][locals()[str(model_list[model_id])+str(trial)+'_rcp85']<0] = 0
        yield_all = np.column_stack((yield_all,locals()[str(model_list[model_id])+str(trial)+'_rcp85']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_rcp85']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1)
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/18
average_model_rcp85 = average_model
production_average_model = np.zeros((1264,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average,model_7_average,model_8_average, model_9_average, model_10_average, model_11_average, model_12_average, model_13_average, model_14_average, model_15_average, model_16_average, model_17_average))
yield_all_model_future_rcp85_m = yield_all
production_all_model = np.zeros((1264,18))
production_all = np.zeros((1264,18000))
for index in range(0,16):
    for year in range(0,79):
        production_all_model[index*79+year,:] = yield_all_model[index*79+year,:]*area[index]
        production_average_model[index*79+year,:] = average_model[index*79+year,:]*area[index]
        production_all[index*79+year,:] = yield_all[index*79+year,:]*area[index]
production_all_split = np.split(production_all, 16)
production_model_split = np.split(production_all_model,16)
production_model_split_85 = production_model_split
production_across_state_rcp85 = np.zeros((79,18))
production_average_model_split = np.split(production_average_model,16)
production_average_model_across_state_rcp85 = np.zeros((79,1000))
production_all_split_across_state = np.zeros((79,18000))
for county_id in range(0,16):
    for year in range(0,79):
        production_across_state_rcp85[year,:] = production_across_state_rcp85[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_rcp85[year,:] = production_average_model_across_state_rcp85[year,:]+production_average_model_split[county_id][year,:]
        production_all_split_across_state[year, :] = production_all_split_across_state[year,:]+production_all_split[county_id][year,:]
production_all_rcp85_m_future = production_all_split_across_state
yield_across_state_future_rcp85_m =np.zeros((79,18))
yield_average_model_future_rcp85_m = np.zeros((79,1000))
yield_all_future_rcp85_m = np.zeros((79,18000))

for year in range(0,79):
    yield_across_state_future_rcp85_m[year,:] = production_across_state_rcp85[year,:]/np.sum(area)
    yield_average_model_future_rcp85_m[year,:] = production_average_model_across_state_rcp85[year,:]/np.sum(area)
    yield_all_future_rcp85_m[year,:] = production_all_split_across_state[year,:]/np.sum(area)

np.save(save_path+'yield_across_state_future_rcp85_m.npy',yield_across_state_future_rcp85_m)
np.save(save_path+'yield_average_model_future_rcp85_m.npy',yield_average_model_future_rcp85_m)
np.save(save_path+'yield_all_future_rcp85_m.npy',yield_all_future_rcp85_m)




np.save(save_path+'yield_all_model_hist_rcp45.npy',yield_all_model_hist_rcp45)

np.save(save_path+'yield_all_model_hist_rcp45_s.npy',yield_all_model_hist_rcp45_s)

np.save(save_path+'yield_all_model_hist_rcp45_m.npy',yield_all_model_hist_rcp45_m)

np.save(save_path+'yield_all_model_hist_rcp85.npy',yield_all_model_hist_rcp85)

np.save(save_path+'yield_all_model_hist_rcp85_s.npy',yield_all_model_hist_rcp85_s)

np.save(save_path+'yield_all_model_hist_rcp85_m.npy',yield_all_model_hist_rcp85_m)

np.save(save_path+'yield_all_model_future_rcp45.npy',yield_all_model_future_rcp45)

np.save(save_path+'yield_all_model_future_rcp45_s.npy',yield_all_model_future_rcp45_s)

np.save(save_path+'yield_all_model_future_rcp45_m.npy',yield_all_model_future_rcp45_m)

np.save(save_path+'yield_all_model_future_rcp85.npy',yield_all_model_future_rcp85)

np.save(save_path+'yield_all_model_future_rcp85_s.npy',yield_all_model_future_rcp85_s)

np.save(save_path+'yield_all_model_future_rcp85_m.npy',yield_all_model_future_rcp85_m)




