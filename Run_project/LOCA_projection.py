##Code to use statistical relationship obtained from Lasso regression between gridMET-ACI and historical almond yield to 
##project almond yield based on LOCA climate datasets for each county, and compute almond cropand area-weighted yield of California. 

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

home_path='/Run_project'
input_path_model = home_path+'/intermediate_data/lasso_model/'
input_path_MACA = home_path+'/intermediate_data/LOCA_csv/'
input_path_area = home_path+'/input_data/'
save_path = home_path+'/output_data/projection/LOCA/'
save_path_projection_csv = home_path+'/intermediate_data/projection_csv/LOCA/'

aci_num = 14
for i in range(1,11):
    locals()['coef'+str(i)] = np.zeros((100,aci_num*2+32))
coef_sum = np.zeros((0,aci_num*2+32))
for i in range(1,1001):
    print(i)
    locals()['coef'+str(((i-1)//100)+1)][i%100-1] = genfromtxt(input_path_model+'coef_'+str(i)+'.csv', delimiter = ',')
for i in range(1,11):
    coef_sum = np.row_stack((coef_sum, locals()['coef'+str(i)]))



county_list = ['Butte', 'Colusa', 'Fresno', 'Glenn', 'Kern', 'Kings', 'Madera', 'Merced', 'San Joaquin', 'Solano', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Yolo', 'Yuba']                      

model_list = ['ACCESS-CM2', 'EC-Earth3', 'EC-Earth3-Veg', 'GFDL-ESM4',  'INM-CM5-0',  'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'CNRM-ESM2-1']

area_csv = genfromtxt(input_path_area+'almond_area.csv', delimiter = ',')


##hist for new model
for trial in range(1,11):
    for model in range(0,8):
        simulation_ssp245 = np.zeros((656,100))
        simulation_ssp245_s = np.zeros((656,100))
        simulation_ssp245_m = np.zeros((656,100))
        aci_ssp245 = genfromtxt(input_path_MACA+str(model_list[model])+'hist_ssp245_ACI.csv', delimiter = ',')
        aci_ssp245_s = genfromtxt(input_path_MACA+str(model_list[model])+'hist_ssp245_s_ACI.csv', delimiter = ',')
        aci_ssp245_m = genfromtxt(input_path_MACA+str(model_list[model])+'hist_ssp245_m_ACI.csv', delimiter = ',')
        for i in range(0,656):
            for j in range(0,100):
                simulation_ssp245[i,j] = np.nansum(aci_ssp245[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_ssp245_s[i,j] = np.nansum(aci_ssp245_s[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_ssp245_m[i,j] = np.nansum(aci_ssp245_m[i,:]*locals()['coef'+str(trial)][j,:])
        savetxt(save_path_projection_csv+str(model_list[model])+'_hist_prediction_'+str(trial)+'_ssp245.csv', simulation_ssp245, delimiter = ',')
        savetxt(save_path_projection_csv+str(model_list[model])+'_hist_prediction_'+str(trial)+'_ssp245_s.csv', simulation_ssp245_s, delimiter = ',')
        savetxt(save_path_projection_csv+str(model_list[model])+'_hist_prediction_'+str(trial)+'_ssp245_m.csv', simulation_ssp245_m, delimiter = ',')

##future
for trial in range(1,11):
    for model in range(0,8):
        simulation_ssp245 = np.zeros((1264,100))
        simulation_ssp245_s = np.zeros((1264,100))
        simulation_ssp245_m = np.zeros((1264,100))
        aci_ssp245 = genfromtxt(input_path_MACA+str(model_list[model])+'future_ssp245_ACI.csv', delimiter = ',')
        aci_ssp245_s = genfromtxt(input_path_MACA+str(model_list[model])+'future_ssp245_s_ACI.csv', delimiter = ',')
        aci_ssp245_m = genfromtxt(input_path_MACA+str(model_list[model])+'future_ssp245_m_ACI.csv', delimiter = ',')
        for i in range(0,1264):
            for j in range(0,100):
                simulation_ssp245[i,j] = np.nansum(aci_ssp245[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_ssp245_s[i,j] = np.nansum(aci_ssp245_s[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_ssp245_m[i,j] = np.nansum(aci_ssp245_m[i,:]*locals()['coef'+str(trial)][j,:])
        savetxt(save_path_projection_csv+str(model_list[model])+'_future_prediction_'+str(trial)+'_ssp245.csv', simulation_ssp245, delimiter = ',')
        savetxt(save_path_projection_csv+str(model_list[model])+'_future_prediction_'+str(trial)+'_ssp245_s.csv', simulation_ssp245_s, delimiter = ',')
        savetxt(save_path_projection_csv+str(model_list[model])+'_future_prediction_'+str(trial)+'_ssp245_m.csv', simulation_ssp245_m, delimiter = ',')




area = area_csv[0:41,:]

yield_all_hist = np.zeros((656,0))
##hist ssp245
average_model = np.zeros((656,1000))
for model_id in range(0,8):
    locals()['model_'+str(model_id)] = np.zeros((656,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_hist'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_hist_prediction_'+str(trial)+'_ssp245.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_hist'][locals()[str(model_list[model_id])+str(trial)+'_hist']<0] = 0
        yield_all_hist = np.column_stack((yield_all_hist,locals()[str(model_list[model_id])+str(trial)+'_hist']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_hist']))
    locals()['model_'+str(model_id)+'_average'] = np.nanmedian(locals()['model_'+str(model_id)], axis = 1) ## change to median 
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/8
average_model_hist = average_model
production_average_model = np.zeros((656,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average, model_7_average))
yield_all_model_hist_ssp245 = yield_all_hist
yield_all_model_hist_ssp245_average_model = yield_all_model
production_all_model = np.zeros((656,8))
production_all_hist = np.zeros((656,8000))
yield_1980_ssp245 = np.zeros((16))
for index in range(0,16):
    yield_1980_ssp245[index] = np.mean(yield_all_model[index * 41,:])
for index in range(0,16):
    for year in range(0,41):
        production_all_model[index*41+year,:] = yield_all_model[index*41+year,:]*area[year,index]
        production_average_model[index*41+year,:] = average_model[index*41+year,:]*area[year,index]
        production_all_hist[index*41+year,:] = yield_all_hist[index*41+year,:]*area[year,index]
production_all_hist_split = np.split(production_all_hist, 16)
production_model_split = np.split(production_all_model,16) 
production_model_split_hist = production_model_split
production_across_state_hist = np.zeros((41,8))
production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_hist = np.zeros((41,1000))
production_all_hist_split_across_state = np.zeros((41,8000)) 
for county_id in range(0,16):
    for year in range(0,41):
        production_across_state_hist[year,:] = production_across_state_hist[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_hist[year,:] = production_average_model_across_state_hist[year,:]+production_average_model_split[county_id][year,:]
        production_all_hist_split_across_state[year, :] = production_all_hist_split_across_state[year,:]+production_all_hist_split[county_id][year,:]
production_all_ssp245_hist = production_all_hist_split_across_state
yield_across_state_hist_ssp245 =np.zeros((41,8))
yield_average_model_hist_ssp245 = np.zeros((41,1000))
yield_all_hist_ssp245 = np.zeros((41,8000))
for year in range(0,41):
    yield_across_state_hist_ssp245[year,:] = production_across_state_hist[year,:]/np.sum(area[year])
    yield_average_model_hist_ssp245[year,:] = production_average_model_across_state_hist[year,:]/np.sum(area[year])
    yield_all_hist_ssp245[year,:] = production_all_hist_split_across_state[year,:]/np.sum(area[year])
np.save(save_path+'yield_across_state_hist_ssp245.npy',yield_across_state_hist_ssp245)
np.save(save_path+'yield_average_model_hist_ssp245.npy',yield_average_model_hist_ssp245)
np.save(save_path+'yield_all_hist_ssp245.npy',yield_all_hist_ssp245)
np.save(save_path+'yield_1980_ssp245.npy',yield_1980_ssp245)



yield_all_hist = np.zeros((656,0))
##hist ssp245
average_model = np.zeros((656,1000))
for model_id in range(0,8):
    locals()['model_'+str(model_id)] = np.zeros((656,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_hist'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_hist_prediction_'+str(trial)+'_ssp245_s.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_hist'][locals()[str(model_list[model_id])+str(trial)+'_hist']<0] = 0
        yield_all_hist = np.column_stack((yield_all_hist,locals()[str(model_list[model_id])+str(trial)+'_hist']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_hist']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1) ## change to median 
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/8
average_model_hist = average_model
production_average_model = np.zeros((656,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average, model_7_average))
yield_all_model_hist_ssp245_s = yield_all_hist
yield_all_model_hist_ssp245_s_average_model = yield_all_model
production_all_model = np.zeros((656,8))
production_all_hist = np.zeros((656,8000))
for index in range(0,16):
    for year in range(0,41):
        production_all_model[index*41+year,:] = yield_all_model[index*41+year,:]*area[year,index]
        production_average_model[index*41+year,:] = average_model[index*41+year,:]*area[year,index]
        production_all_hist[index*41+year,:] = yield_all_hist[index*41+year,:]*area[year,index]

production_all_hist_split = np.split(production_all_hist, 16)
production_model_split = np.split(production_all_model,16) 
production_model_split_hist = production_model_split
production_across_state_hist = np.zeros((41,8))
production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_hist = np.zeros((41,1000))
production_all_hist_split_across_state = np.zeros((41,8000))
for county_id in range(0,16):
    for year in range(0,41):
        production_across_state_hist[year,:] = production_across_state_hist[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_hist[year,:] = production_average_model_across_state_hist[year,:]+production_average_model_split[county_id][year,:]
        production_all_hist_split_across_state[year, :] = production_all_hist_split_across_state[year,:]+production_all_hist_split[county_id][year,:]
production_all_ssp245_s_hist = production_all_hist_split_across_state
yield_across_state_hist_ssp245_s =np.zeros((41,8))
yield_average_model_hist_ssp245_s = np.zeros((41,1000))
yield_all_hist_ssp245_s = np.zeros((41,8000))
for year in range(0,41):
    yield_across_state_hist_ssp245_s[year,:] = production_across_state_hist[year,:]/np.sum(area[year])
    yield_average_model_hist_ssp245_s[year,:] = production_average_model_across_state_hist[year,:]/np.sum(area[year])
    yield_all_hist_ssp245_s[year,:] = production_all_hist_split_across_state[year,:]/np.sum(area[year])
np.save(save_path+'yield_across_state_hist_ssp245_s.npy',yield_across_state_hist_ssp245_s)
np.save(save_path+'yield_average_model_hist_ssp245_s.npy',yield_average_model_hist_ssp245_s)
np.save(save_path+'yield_all_hist_ssp245_s.npy',yield_all_hist_ssp245_s)


yield_all_hist = np.zeros((656,0))
##hist ssp245_m
average_model = np.zeros((656,1000))
for model_id in range(0,8):
    locals()['model_'+str(model_id)] = np.zeros((656,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_hist'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_hist_prediction_'+str(trial)+'_ssp245_m.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_hist'][locals()[str(model_list[model_id])+str(trial)+'_hist']<0] = 0
        yield_all_hist = np.column_stack((yield_all_hist,locals()[str(model_list[model_id])+str(trial)+'_hist']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_hist']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1) ## change to median
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/8
average_model_hist = average_model
production_average_model = np.zeros((656,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average, model_7_average))
yield_all_model_hist_ssp245_m = yield_all_hist
yield_all_model_hist_ssp245_m_average_model = yield_all_model
production_all_model = np.zeros((656,8))
production_all_hist = np.zeros((656,8000))
for index in range(0,16):
    for year in range(0,41):
        production_all_model[index*41+year,:] = yield_all_model[index*41+year,:]*area[year,index]
        production_average_model[index*41+year,:] = average_model[index*41+year,:]*area[year,index]
        production_all_hist[index*41+year,:] = yield_all_hist[index*41+year,:]*area[year,index]

production_all_hist_split = np.split(production_all_hist, 16)
production_model_split = np.split(production_all_model,16)
production_model_split_hist = production_model_split
production_across_state_hist = np.zeros((41,8))
production_average_model_split = np.split(production_average_model,16)
production_average_model_across_state_hist = np.zeros((41,1000))
production_all_hist_split_across_state = np.zeros((41,8000))
for county_id in range(0,16):
    for year in range(0,41):
        production_across_state_hist[year,:] = production_across_state_hist[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_hist[year,:] = production_average_model_across_state_hist[year,:]+production_average_model_split[county_id][year,:]
        production_all_hist_split_across_state[year, :] = production_all_hist_split_across_state[year,:]+production_all_hist_split[county_id][year,:]
production_all_ssp245_m_hist = production_all_hist_split_across_state
yield_across_state_hist_ssp245_m =np.zeros((41,8))
yield_average_model_hist_ssp245_m = np.zeros((41,1000))
yield_all_hist_ssp245_m = np.zeros((41,8000))
for year in range(0,41):
    yield_across_state_hist_ssp245_m[year,:] = production_across_state_hist[year,:]/np.sum(area[year])
    yield_average_model_hist_ssp245_m[year,:] = production_average_model_across_state_hist[year,:]/np.sum(area[year])
    yield_all_hist_ssp245_m[year,:] = production_all_hist_split_across_state[year,:]/np.sum(area[year])
np.save(save_path+'yield_across_state_hist_ssp245_m.npy',yield_across_state_hist_ssp245_m)
np.save(save_path+'yield_average_model_hist_ssp245_m.npy',yield_average_model_hist_ssp245_m)
np.save(save_path+'yield_all_hist_ssp245_m.npy',yield_all_hist_ssp245_m)





area = area[-1]
#future ssp245
yield_all = np.zeros((1264,0))
average_model = np.zeros((1264,1000))
for model_id in range(0,8):
    locals()['model_'+str(model_id)] = np.zeros((1264,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_ssp245'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_future_prediction_'+str(trial)+'_ssp245.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_ssp245'][locals()[str(model_list[model_id])+str(trial)+'_ssp245']<0] = 0
        yield_all = np.column_stack((yield_all,locals()[str(model_list[model_id])+str(trial)+'_ssp245']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_ssp245']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1)
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/8
average_model_ssp245 = average_model
production_average_model = np.zeros((1264,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average, model_7_average))
yield_all_model_future_ssp245 = yield_all
production_all_model = np.zeros((1264,8))
production_all = np.zeros((1264,8000))
for index in range(0,16):
    for year in range(0,79):
        production_all_model[index*79+year,:] = yield_all_model[index*79+year,:]*area[index]
        production_average_model[index*79+year,:] = average_model[index*79+year,:]*area[index]
        production_all[index*79+year,:] = yield_all[index*79+year,:]*area[index]
production_all_split = np.split(production_all, 16)
production_model_split = np.split(production_all_model,16) 
production_model_split_45 = production_model_split
production_across_state_ssp245 = np.zeros((79,8))
production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_ssp245 = np.zeros((79,1000))
production_all_split_across_state = np.zeros((79,8000))
for county_id in range(0,16):
    for year in range(0,79):
        production_across_state_ssp245[year,:] = production_across_state_ssp245[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_ssp245[year,:] = production_average_model_across_state_ssp245[year,:]+production_average_model_split[county_id][year,:]
        production_all_split_across_state[year, :] = production_all_split_across_state[year,:]+production_all_split[county_id][year,:]
production_all_ssp245_future = production_all_split_across_state
yield_across_state_future_ssp245 =np.zeros((79,8))
yield_average_model_future_ssp245 = np.zeros((79,1000))
yield_all_future_ssp245 = np.zeros((79,8000))

for year in range(0,79):
    yield_across_state_future_ssp245[year,:] = production_across_state_ssp245[year,:]/np.sum(area)
    yield_average_model_future_ssp245[year,:] = production_average_model_across_state_ssp245[year,:]/np.sum(area)
    yield_all_future_ssp245[year,:] = production_all_split_across_state[year,:]/np.sum(area)
np.save(save_path+'yield_across_state_future_ssp245.npy',yield_across_state_future_ssp245)
np.save(save_path+'yield_average_model_future_ssp245.npy',yield_average_model_future_ssp245)
np.save(save_path+'yield_all_future_ssp245.npy',yield_all_future_ssp245)


# future ssp245_s
yield_all = np.zeros((1264,0))
average_model = np.zeros((1264,1000))
for model_id in range(0,8):
    print(model_id)
    locals()['model_'+str(model_id)] = np.zeros((1264,0))
    for trial in range(1,11):
        print(trial)
        locals()[str(model_list[model_id])+str(trial)+'_ssp245'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_future_prediction_'+str(trial)+'_ssp245_s.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_ssp245'][locals()[str(model_list[model_id])+str(trial)+'_ssp245']<0] = 0
        yield_all = np.column_stack((yield_all,locals()[str(model_list[model_id])+str(trial)+'_ssp245']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_ssp245']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1)
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/8
average_model_ssp245 = average_model
production_average_model = np.zeros((1264,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average, model_7_average))
yield_all_model_future_ssp245_s = yield_all
production_all_model = np.zeros((1264,8))
production_all = np.zeros((1264,8000))
for index in range(0,16):
    for year in range(0,79):
        production_all_model[index*79+year,:] = yield_all_model[index*79+year,:]*area[index]
        production_average_model[index*79+year,:] = average_model[index*79+year,:]*area[index]
        production_all[index*79+year,:] = yield_all[index*79+year,:]*area[index]
production_all_split = np.split(production_all, 16)
production_model_split = np.split(production_all_model,16) 
production_model_split_45 = production_model_split
production_across_state_ssp245 = np.zeros((79,8))
production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_ssp245 = np.zeros((79,1000))
production_all_split_across_state = np.zeros((79,8000))
for county_id in range(0,16):
    for year in range(0,79):
        production_across_state_ssp245[year,:] = production_across_state_ssp245[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_ssp245[year,:] = production_average_model_across_state_ssp245[year,:]+production_average_model_split[county_id][year,:]
        production_all_split_across_state[year, :] = production_all_split_across_state[year,:]+production_all_split[county_id][year,:]
production_all_ssp245_s_future = production_all_split_across_state
yield_across_state_future_ssp245_s =np.zeros((79,8))
yield_average_model_future_ssp245_s = np.zeros((79,1000))
yield_all_future_ssp245_s = np.zeros((79,8000))

for year in range(0,79):
    yield_across_state_future_ssp245_s[year,:] = production_across_state_ssp245[year,:]/np.sum(area)
    yield_average_model_future_ssp245_s[year,:] = production_average_model_across_state_ssp245[year,:]/np.sum(area)
    yield_all_future_ssp245_s[year,:] = production_all_split_across_state[year,:]/np.sum(area)
np.save(save_path+'yield_across_state_future_ssp245_s.npy',yield_across_state_future_ssp245_s)
np.save(save_path+'yield_average_model_future_ssp245_s.npy',yield_average_model_future_ssp245_s)
np.save(save_path+'yield_all_future_ssp245_s.npy',yield_all_future_ssp245_s)

# future ssp245_m
yield_all = np.zeros((1264,0))
average_model = np.zeros((1264,1000))
for model_id in range(0,8):
    print(model_id)
    locals()['model_'+str(model_id)] = np.zeros((1264,0))
    for trial in range(1,11):
        print(trial)
        locals()[str(model_list[model_id])+str(trial)+'_ssp245'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_future_prediction_'+str(trial)+'_ssp245_m.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_ssp245'][locals()[str(model_list[model_id])+str(trial)+'_ssp245']<0] = 0
        yield_all = np.column_stack((yield_all,locals()[str(model_list[model_id])+str(trial)+'_ssp245']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_ssp245']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1)
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/8
average_model_ssp245 = average_model
production_average_model = np.zeros((1264,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average, model_7_average))
yield_all_model_future_ssp245_m = yield_all
production_all_model = np.zeros((1264,8))
production_all = np.zeros((1264,8000))
for index in range(0,16):
    for year in range(0,79):
        production_all_model[index*79+year,:] = yield_all_model[index*79+year,:]*area[index]
        production_average_model[index*79+year,:] = average_model[index*79+year,:]*area[index]
        production_all[index*79+year,:] = yield_all[index*79+year,:]*area[index]
production_all_split = np.split(production_all, 16)
production_model_split = np.split(production_all_model,16)
production_model_split_45 = production_model_split
production_across_state_ssp245 = np.zeros((79,8))
production_average_model_split = np.split(production_average_model,16)
production_average_model_across_state_ssp245 = np.zeros((79,1000))
production_all_split_across_state = np.zeros((79,8000))
for county_id in range(0,16):
    for year in range(0,79):
        production_across_state_ssp245[year,:] = production_across_state_ssp245[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_ssp245[year,:] = production_average_model_across_state_ssp245[year,:]+production_average_model_split[county_id][year,:]
        production_all_split_across_state[year, :] = production_all_split_across_state[year,:]+production_all_split[county_id][year,:]
production_all_ssp245_m_future = production_all_split_across_state
yield_across_state_future_ssp245_m =np.zeros((79,8))
yield_average_model_future_ssp245_m = np.zeros((79,1000))
yield_all_future_ssp245_m = np.zeros((79,8000))

for year in range(0,79):
    yield_across_state_future_ssp245_m[year,:] = production_across_state_ssp245[year,:]/np.sum(area)
    yield_average_model_future_ssp245_m[year,:] = production_average_model_across_state_ssp245[year,:]/np.sum(area)
    yield_all_future_ssp245_m[year,:] = production_all_split_across_state[year,:]/np.sum(area)
np.save(save_path+'yield_across_state_future_ssp245_m.npy',yield_across_state_future_ssp245_m)
np.save(save_path+'yield_average_model_future_ssp245_m.npy',yield_average_model_future_ssp245_m)
np.save(save_path+'yield_all_future_ssp245_m.npy',yield_all_future_ssp245_m)



np.save(save_path+'yield_all_model_hist_ssp245.npy',yield_all_model_hist_ssp245)

np.save(save_path+'yield_all_model_hist_ssp245_s.npy',yield_all_model_hist_ssp245_s)

np.save(save_path+'yield_all_model_hist_ssp245_m.npy',yield_all_model_hist_ssp245_m)


np.save(save_path+'yield_all_model_future_ssp245.npy',yield_all_model_future_ssp245)

np.save(save_path+'yield_all_model_future_ssp245_s.npy',yield_all_model_future_ssp245_s)

np.save(save_path+'yield_all_model_future_ssp245_m.npy',yield_all_model_future_ssp245_m)



##hist for new model
for trial in range(1,11):
    for model in range(0,8):
        simulation_ssp585 = np.zeros((656,100))
        simulation_ssp585_s = np.zeros((656,100))
        simulation_ssp585_m = np.zeros((656,100))
        aci_ssp585 = genfromtxt(input_path_MACA+str(model_list[model])+'hist_ssp585_ACI.csv', delimiter = ',')
        aci_ssp585_s = genfromtxt(input_path_MACA+str(model_list[model])+'hist_ssp585_s_ACI.csv', delimiter = ',')
        aci_ssp585_m = genfromtxt(input_path_MACA+str(model_list[model])+'hist_ssp585_m_ACI.csv', delimiter = ',')
        for i in range(0,656):
            for j in range(0,100):
                simulation_ssp585[i,j] = np.nansum(aci_ssp585[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_ssp585_s[i,j] = np.nansum(aci_ssp585_s[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_ssp585_m[i,j] = np.nansum(aci_ssp585_m[i,:]*locals()['coef'+str(trial)][j,:])
        savetxt(save_path_projection_csv+str(model_list[model])+'_hist_prediction_'+str(trial)+'_ssp585.csv', simulation_ssp585, delimiter = ',')
        savetxt(save_path_projection_csv+str(model_list[model])+'_hist_prediction_'+str(trial)+'_ssp585_s.csv', simulation_ssp585_s, delimiter = ',')
        savetxt(save_path_projection_csv+str(model_list[model])+'_hist_prediction_'+str(trial)+'_ssp585_m.csv', simulation_ssp585_m, delimiter = ',')

##future
for trial in range(1,11):
    for model in range(0,8):
        simulation_ssp585 = np.zeros((1264,100))
        simulation_ssp585_s = np.zeros((1264,100))
        simulation_ssp585_m = np.zeros((1264,100))
        aci_ssp585 = genfromtxt(input_path_MACA+str(model_list[model])+'future_ssp585_ACI.csv', delimiter = ',')
        aci_ssp585_s = genfromtxt(input_path_MACA+str(model_list[model])+'future_ssp585_s_ACI.csv', delimiter = ',')
        aci_ssp585_m = genfromtxt(input_path_MACA+str(model_list[model])+'future_ssp585_m_ACI.csv', delimiter = ',')
        for i in range(0,1264):
            for j in range(0,100):
                simulation_ssp585[i,j] = np.nansum(aci_ssp585[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_ssp585_s[i,j] = np.nansum(aci_ssp585_s[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_ssp585_m[i,j] = np.nansum(aci_ssp585_m[i,:]*locals()['coef'+str(trial)][j,:])
        savetxt(save_path_projection_csv+str(model_list[model])+'_future_prediction_'+str(trial)+'_ssp585.csv', simulation_ssp585, delimiter = ',')
        savetxt(save_path_projection_csv+str(model_list[model])+'_future_prediction_'+str(trial)+'_ssp585_s.csv', simulation_ssp585_s, delimiter = ',')
        savetxt(save_path_projection_csv+str(model_list[model])+'_future_prediction_'+str(trial)+'_ssp585_m.csv', simulation_ssp585_m, delimiter = ',')




area = area_csv[0:41,:]

yield_all_hist = np.zeros((656,0))
##hist ssp585
average_model = np.zeros((656,1000))
for model_id in range(0,8):
    locals()['model_'+str(model_id)] = np.zeros((656,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_hist'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_hist_prediction_'+str(trial)+'_ssp585.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_hist'][locals()[str(model_list[model_id])+str(trial)+'_hist']<0] = 0
        yield_all_hist = np.column_stack((yield_all_hist,locals()[str(model_list[model_id])+str(trial)+'_hist']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_hist']))
    locals()['model_'+str(model_id)+'_average'] = np.nanmedian(locals()['model_'+str(model_id)], axis = 1) ## change to median
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/8
average_model_hist = average_model
production_average_model = np.zeros((656,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average, model_7_average))
yield_all_model_hist_ssp585 = yield_all_hist
yield_all_model_hist_ssp585_average_model = yield_all_model
production_all_model = np.zeros((656,8))
production_all_hist = np.zeros((656,8000))
yield_1980_ssp585 = np.zeros((16))
for index in range(0,16):
    yield_1980_ssp585[index] = np.mean(yield_all_model[index * 41,:])
for index in range(0,16):
    for year in range(0,41):
        production_all_model[index*41+year,:] = yield_all_model[index*41+year,:]*area[year,index]
        production_average_model[index*41+year,:] = average_model[index*41+year,:]*area[year,index]
        production_all_hist[index*41+year,:] = yield_all_hist[index*41+year,:]*area[year,index]
production_all_hist_split = np.split(production_all_hist, 16)
production_model_split = np.split(production_all_model,16)
production_model_split_hist = production_model_split
production_across_state_hist = np.zeros((41,8))
production_average_model_split = np.split(production_average_model,16)
production_average_model_across_state_hist = np.zeros((41,1000))
production_all_hist_split_across_state = np.zeros((41,8000))
for county_id in range(0,16):
    for year in range(0,41):
        production_across_state_hist[year,:] = production_across_state_hist[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_hist[year,:] = production_average_model_across_state_hist[year,:]+production_average_model_split[county_id][year,:]
        production_all_hist_split_across_state[year, :] = production_all_hist_split_across_state[year,:]+production_all_hist_split[county_id][year,:]
production_all_ssp585_hist = production_all_hist_split_across_state
yield_across_state_hist_ssp585 =np.zeros((41,8))
yield_average_model_hist_ssp585 = np.zeros((41,1000))
yield_all_hist_ssp585 = np.zeros((41,8000))
for year in range(0,41):
    yield_across_state_hist_ssp585[year,:] = production_across_state_hist[year,:]/np.sum(area[year])
    yield_average_model_hist_ssp585[year,:] = production_average_model_across_state_hist[year,:]/np.sum(area[year])
    yield_all_hist_ssp585[year,:] = production_all_hist_split_across_state[year,:]/np.sum(area[year])
np.save(save_path+'yield_across_state_hist_ssp585.npy',yield_across_state_hist_ssp585)
np.save(save_path+'yield_average_model_hist_ssp585.npy',yield_average_model_hist_ssp585)
np.save(save_path+'yield_all_hist_ssp585.npy',yield_all_hist_ssp585)
np.save(save_path+'yield_1980_ssp585.npy',yield_1980_ssp585)


yield_all_hist = np.zeros((656,0))
##hist ssp585
average_model = np.zeros((656,1000))
for model_id in range(0,8):
    locals()['model_'+str(model_id)] = np.zeros((656,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_hist'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_hist_prediction_'+str(trial)+'_ssp585_s.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_hist'][locals()[str(model_list[model_id])+str(trial)+'_hist']<0] = 0
        yield_all_hist = np.column_stack((yield_all_hist,locals()[str(model_list[model_id])+str(trial)+'_hist']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_hist']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1) ## change to median
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/8
average_model_hist = average_model
production_average_model = np.zeros((656,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average, model_7_average))
yield_all_model_hist_ssp585_s = yield_all_hist
yield_all_model_hist_ssp585_s_average_model = yield_all_model
production_all_model = np.zeros((656,8))
production_all_hist = np.zeros((656,8000))
for index in range(0,16):
    for year in range(0,41):
        production_all_model[index*41+year,:] = yield_all_model[index*41+year,:]*area[year,index]
        production_average_model[index*41+year,:] = average_model[index*41+year,:]*area[year,index]
        production_all_hist[index*41+year,:] = yield_all_hist[index*41+year,:]*area[year,index]

production_all_hist_split = np.split(production_all_hist, 16)
production_model_split = np.split(production_all_model,16)
production_model_split_hist = production_model_split
production_across_state_hist = np.zeros((41,8))
production_average_model_split = np.split(production_average_model,16)
production_average_model_across_state_hist = np.zeros((41,1000))
production_all_hist_split_across_state = np.zeros((41,8000))
for county_id in range(0,16):
    for year in range(0,41):
        production_across_state_hist[year,:] = production_across_state_hist[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_hist[year,:] = production_average_model_across_state_hist[year,:]+production_average_model_split[county_id][year,:]
        production_all_hist_split_across_state[year, :] = production_all_hist_split_across_state[year,:]+production_all_hist_split[county_id][year,:]
production_all_ssp585_s_hist = production_all_hist_split_across_state
yield_across_state_hist_ssp585_s =np.zeros((41,8))
yield_average_model_hist_ssp585_s = np.zeros((41,1000))
yield_all_hist_ssp585_s = np.zeros((41,8000))
for year in range(0,41):
    yield_across_state_hist_ssp585_s[year,:] = production_across_state_hist[year,:]/np.sum(area[year])
    yield_average_model_hist_ssp585_s[year,:] = production_average_model_across_state_hist[year,:]/np.sum(area[year])
    yield_all_hist_ssp585_s[year,:] = production_all_hist_split_across_state[year,:]/np.sum(area[year])
np.save(save_path+'yield_across_state_hist_ssp585_s.npy',yield_across_state_hist_ssp585_s)
np.save(save_path+'yield_average_model_hist_ssp585_s.npy',yield_average_model_hist_ssp585_s)
np.save(save_path+'yield_all_hist_ssp585_s.npy',yield_all_hist_ssp585_s)


yield_all_hist = np.zeros((656,0))
##hist ssp585_m
average_model = np.zeros((656,1000))
for model_id in range(0,8):
    locals()['model_'+str(model_id)] = np.zeros((656,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_hist'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_hist_prediction_'+str(trial)+'_ssp585_m.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_hist'][locals()[str(model_list[model_id])+str(trial)+'_hist']<0] = 0
        yield_all_hist = np.column_stack((yield_all_hist,locals()[str(model_list[model_id])+str(trial)+'_hist']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_hist']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1) ## change to median
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/8
average_model_hist = average_model
production_average_model = np.zeros((656,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average, model_7_average))
yield_all_model_hist_ssp585_m = yield_all_hist
yield_all_model_hist_ssp585_m_average_model = yield_all_model
production_all_model = np.zeros((656,8))
production_all_hist = np.zeros((656,8000))
for index in range(0,16):
    for year in range(0,41):
        production_all_model[index*41+year,:] = yield_all_model[index*41+year,:]*area[year,index]
        production_average_model[index*41+year,:] = average_model[index*41+year,:]*area[year,index]
        production_all_hist[index*41+year,:] = yield_all_hist[index*41+year,:]*area[year,index]

production_all_hist_split = np.split(production_all_hist, 16)
production_model_split = np.split(production_all_model,16)
production_model_split_hist = production_model_split
production_across_state_hist = np.zeros((41,8))
production_average_model_split = np.split(production_average_model,16)
production_average_model_across_state_hist = np.zeros((41,1000))
production_all_hist_split_across_state = np.zeros((41,8000))
for county_id in range(0,16):
    for year in range(0,41):
        production_across_state_hist[year,:] = production_across_state_hist[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_hist[year,:] = production_average_model_across_state_hist[year,:]+production_average_model_split[county_id][year,:]
        production_all_hist_split_across_state[year, :] = production_all_hist_split_across_state[year,:]+production_all_hist_split[county_id][year,:]
production_all_ssp585_m_hist = production_all_hist_split_across_state
yield_across_state_hist_ssp585_m =np.zeros((41,8))
yield_average_model_hist_ssp585_m = np.zeros((41,1000))
yield_all_hist_ssp585_m = np.zeros((41,8000))
for year in range(0,41):
    yield_across_state_hist_ssp585_m[year,:] = production_across_state_hist[year,:]/np.sum(area[year])
    yield_average_model_hist_ssp585_m[year,:] = production_average_model_across_state_hist[year,:]/np.sum(area[year])
    yield_all_hist_ssp585_m[year,:] = production_all_hist_split_across_state[year,:]/np.sum(area[year])
np.save(save_path+'yield_across_state_hist_ssp585_m.npy',yield_across_state_hist_ssp585_m)
np.save(save_path+'yield_average_model_hist_ssp585_m.npy',yield_average_model_hist_ssp585_m)
np.save(save_path+'yield_all_hist_ssp585_m.npy',yield_all_hist_ssp585_m)


area = area[-1]
#future ssp585
yield_all = np.zeros((1264,0))
average_model = np.zeros((1264,1000))
for model_id in range(0,8):
    locals()['model_'+str(model_id)] = np.zeros((1264,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_ssp585'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_future_prediction_'+str(trial)+'_ssp585.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_ssp585'][locals()[str(model_list[model_id])+str(trial)+'_ssp585']<0] = 0
        yield_all = np.column_stack((yield_all,locals()[str(model_list[model_id])+str(trial)+'_ssp585']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_ssp585']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1)
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/8
average_model_ssp585 = average_model
production_average_model = np.zeros((1264,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average, model_7_average))
yield_all_model_future_ssp585 = yield_all
production_all_model = np.zeros((1264,8))
production_all = np.zeros((1264,8000))
for index in range(0,16):
    for year in range(0,79):
        production_all_model[index*79+year,:] = yield_all_model[index*79+year,:]*area[index]
        production_average_model[index*79+year,:] = average_model[index*79+year,:]*area[index]
        production_all[index*79+year,:] = yield_all[index*79+year,:]*area[index]
production_all_split = np.split(production_all, 16)
production_model_split = np.split(production_all_model,16)
production_model_split_45 = production_model_split
production_across_state_ssp585 = np.zeros((79,8))
production_average_model_split = np.split(production_average_model,16)
production_average_model_across_state_ssp585 = np.zeros((79,1000))
production_all_split_across_state = np.zeros((79,8000))
for county_id in range(0,16):
    for year in range(0,79):
        production_across_state_ssp585[year,:] = production_across_state_ssp585[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_ssp585[year,:] = production_average_model_across_state_ssp585[year,:]+production_average_model_split[county_id][year,:]
        production_all_split_across_state[year, :] = production_all_split_across_state[year,:]+production_all_split[county_id][year,:]
production_all_ssp585_future = production_all_split_across_state
yield_across_state_future_ssp585 =np.zeros((79,8))
yield_average_model_future_ssp585 = np.zeros((79,1000))
yield_all_future_ssp585 = np.zeros((79,8000))

for year in range(0,79):
    yield_across_state_future_ssp585[year,:] = production_across_state_ssp585[year,:]/np.sum(area)
    yield_average_model_future_ssp585[year,:] = production_average_model_across_state_ssp585[year,:]/np.sum(area)
    yield_all_future_ssp585[year,:] = production_all_split_across_state[year,:]/np.sum(area)
np.save(save_path+'yield_across_state_future_ssp585.npy',yield_across_state_future_ssp585)
np.save(save_path+'yield_average_model_future_ssp585.npy',yield_average_model_future_ssp585)
np.save(save_path+'yield_all_future_ssp585.npy',yield_all_future_ssp585)


# future ssp585_s
yield_all = np.zeros((1264,0))
average_model = np.zeros((1264,1000))
for model_id in range(0,8):
    print(model_id)
    locals()['model_'+str(model_id)] = np.zeros((1264,0))
    for trial in range(1,11):
        print(trial)
        locals()[str(model_list[model_id])+str(trial)+'_ssp585'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_future_prediction_'+str(trial)+'_ssp585_s.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_ssp585'][locals()[str(model_list[model_id])+str(trial)+'_ssp585']<0] = 0
        yield_all = np.column_stack((yield_all,locals()[str(model_list[model_id])+str(trial)+'_ssp585']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_ssp585']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1)
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/8
average_model_ssp585 = average_model
production_average_model = np.zeros((1264,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average, model_7_average))
yield_all_model_future_ssp585_s = yield_all
production_all_model = np.zeros((1264,8))
production_all = np.zeros((1264,8000))
for index in range(0,16):
    for year in range(0,79):
        production_all_model[index*79+year,:] = yield_all_model[index*79+year,:]*area[index]
        production_average_model[index*79+year,:] = average_model[index*79+year,:]*area[index]
        production_all[index*79+year,:] = yield_all[index*79+year,:]*area[index]
production_all_split = np.split(production_all, 16)
production_model_split = np.split(production_all_model,16)
production_model_split_45 = production_model_split
production_across_state_ssp585 = np.zeros((79,8))
production_average_model_split = np.split(production_average_model,16)
production_average_model_across_state_ssp585 = np.zeros((79,1000))
production_all_split_across_state = np.zeros((79,8000))
for county_id in range(0,16):
    for year in range(0,79):
        production_across_state_ssp585[year,:] = production_across_state_ssp585[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_ssp585[year,:] = production_average_model_across_state_ssp585[year,:]+production_average_model_split[county_id][year,:]
        production_all_split_across_state[year, :] = production_all_split_across_state[year,:]+production_all_split[county_id][year,:]
production_all_ssp585_s_future = production_all_split_across_state
yield_across_state_future_ssp585_s =np.zeros((79,8))
yield_average_model_future_ssp585_s = np.zeros((79,1000))
yield_all_future_ssp585_s = np.zeros((79,8000))

for year in range(0,79):
    yield_across_state_future_ssp585_s[year,:] = production_across_state_ssp585[year,:]/np.sum(area)
    yield_average_model_future_ssp585_s[year,:] = production_average_model_across_state_ssp585[year,:]/np.sum(area)
    yield_all_future_ssp585_s[year,:] = production_all_split_across_state[year,:]/np.sum(area)
np.save(save_path+'yield_across_state_future_ssp585_s.npy',yield_across_state_future_ssp585_s)
np.save(save_path+'yield_average_model_future_ssp585_s.npy',yield_average_model_future_ssp585_s)
np.save(save_path+'yield_all_future_ssp585_s.npy',yield_all_future_ssp585_s)


# future ssp585_m
yield_all = np.zeros((1264,0))
average_model = np.zeros((1264,1000))
for model_id in range(0,8):
    print(model_id)
    locals()['model_'+str(model_id)] = np.zeros((1264,0))
    for trial in range(1,11):
        print(trial)
        locals()[str(model_list[model_id])+str(trial)+'_ssp585'] = genfromtxt(save_path_projection_csv+str(model_list[model_id])+'_future_prediction_'+str(trial)+'_ssp585_m.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_ssp585'][locals()[str(model_list[model_id])+str(trial)+'_ssp585']<0] = 0
        yield_all = np.column_stack((yield_all,locals()[str(model_list[model_id])+str(trial)+'_ssp585']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_ssp585']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1)
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/8
average_model_ssp585 = average_model
production_average_model = np.zeros((1264,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average, model_7_average))
yield_all_model_future_ssp585_m = yield_all
production_all_model = np.zeros((1264,8))
production_all = np.zeros((1264,8000))
for index in range(0,16):
    for year in range(0,79):
        production_all_model[index*79+year,:] = yield_all_model[index*79+year,:]*area[index]
        production_average_model[index*79+year,:] = average_model[index*79+year,:]*area[index]
        production_all[index*79+year,:] = yield_all[index*79+year,:]*area[index]
production_all_split = np.split(production_all, 16)
production_model_split = np.split(production_all_model,16)
production_model_split_45 = production_model_split
production_across_state_ssp585 = np.zeros((79,8))
production_average_model_split = np.split(production_average_model,16)
production_average_model_across_state_ssp585 = np.zeros((79,1000))
production_all_split_across_state = np.zeros((79,8000))
for county_id in range(0,16):
    for year in range(0,79):
        production_across_state_ssp585[year,:] = production_across_state_ssp585[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_ssp585[year,:] = production_average_model_across_state_ssp585[year,:]+production_average_model_split[county_id][year,:]
        production_all_split_across_state[year, :] = production_all_split_across_state[year,:]+production_all_split[county_id][year,:]
production_all_ssp585_m_future = production_all_split_across_state
yield_across_state_future_ssp585_m =np.zeros((79,8))
yield_average_model_future_ssp585_m = np.zeros((79,1000))
yield_all_future_ssp585_m = np.zeros((79,8000))

for year in range(0,79):
    yield_across_state_future_ssp585_m[year,:] = production_across_state_ssp585[year,:]/np.sum(area)
    yield_average_model_future_ssp585_m[year,:] = production_average_model_across_state_ssp585[year,:]/np.sum(area)
    yield_all_future_ssp585_m[year,:] = production_all_split_across_state[year,:]/np.sum(area)
np.save(save_path+'yield_across_state_future_ssp585_m.npy',yield_across_state_future_ssp585_m)
np.save(save_path+'yield_average_model_future_ssp585_m.npy',yield_average_model_future_ssp585_m)
np.save(save_path+'yield_all_future_ssp585_m.npy',yield_all_future_ssp585_m)



np.save(save_path+'yield_all_model_hist_ssp585.npy',yield_all_model_hist_ssp585)

np.save(save_path+'yield_all_model_hist_ssp585_s.npy',yield_all_model_hist_ssp585_s)

np.save(save_path+'yield_all_model_hist_ssp585_m.npy',yield_all_model_hist_ssp585_m)


np.save(save_path+'yield_all_model_future_ssp585.npy',yield_all_model_future_ssp585)

np.save(save_path+'yield_all_model_future_ssp585_s.npy',yield_all_model_future_ssp585_s)

np.save(save_path+'yield_all_model_future_ssp585_m.npy',yield_all_model_future_ssp585_m)

