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
from matplotlib.legend_handler import HandlerTuple
import matplotlib.legend_handler
import geopandas
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.font_manager as font_manager
import matplotlib.patheffects as path_effects
from matplotlib import gridspec
from scipy.interpolate import make_interp_spline
from matplotlib.patches import PathPatch
from matplotlib import cm
from matplotlib import gridspec

home_path='~/Run_project'
input_path_gridmet = home_path+'/intermediate_data/Gridmet_csv/'
input_path_projection = home_path+'/output_data/projection/LOCA/'
input_path_contribution = home_path+'/output_data/aci_contribution/LOCA/'
input_path_model = home_path+'/intermediate_data/lasso_model/'
input_path = home_path+'/input_data/'
save_path = home_path+'/output_data/plots/'
shp_path = home_path+'/input_data/CA_Counties/'
input_path_LOCA = home_path+'/intermediate_data/LOCA_csv/'
input_path_projection_MACA = home_path+'/output_data/projection/MACA/'
input_path_contribution_MACA = home_path+'/output_data/aci_contribution/MACA/'
input_path_MACA = home_path+'/intermediate_data/MACA_csv/'

county_list = ['Butte', 'Colusa', 'Fresno', 'Glenn', 'Kern', 'Kings', 'Madera', 'Merced', 'San Joaquin', 'Solano', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Yolo', 'Yuba']                      

## Load yield simulations
yield_all_future_ssp245 = np.load(input_path_projection+'yield_all_future_ssp245.npy')
yield_all_future_ssp245_s = np.load(input_path_projection+'yield_all_future_ssp245_s.npy')
yield_all_future_ssp245_m = np.load(input_path_projection+'yield_all_future_ssp245_m.npy')
yield_all_future_ssp585 = np.load(input_path_projection+'yield_all_future_ssp585.npy')
yield_all_future_ssp585_s = np.load(input_path_projection+'yield_all_future_ssp585_s.npy')
yield_all_future_ssp585_m = np.load(input_path_projection+'yield_all_future_ssp585_m.npy')
yield_all_hist_ssp245 = np.load(input_path_projection+'yield_all_hist_ssp245.npy')
yield_all_hist_ssp245_s = np.load(input_path_projection+'yield_all_hist_ssp245_s.npy')
yield_all_hist_ssp245_m = np.load(input_path_projection+'yield_all_hist_ssp245_m.npy')
yield_all_hist_ssp585 = np.load(input_path_projection+'yield_all_hist_ssp585.npy')
yield_all_hist_ssp585_s = np.load(input_path_projection+'yield_all_hist_ssp585_s.npy')
yield_all_hist_ssp585_m = np.load(input_path_projection+'yield_all_hist_ssp585_m.npy')

yield_across_state_hist_ssp245 = np.load(input_path_projection+'yield_across_state_hist_ssp245.npy')
yield_across_state_hist_ssp245_s = np.load(input_path_projection+'yield_across_state_hist_ssp245_s.npy')
yield_across_state_hist_ssp585 = np.load(input_path_projection+'yield_across_state_hist_ssp585.npy')
yield_across_state_hist_ssp585_s = np.load(input_path_projection+'yield_across_state_hist_ssp585_s.npy')
yield_across_state_future_ssp245 = np.load(input_path_projection+'yield_across_state_future_ssp245.npy')
yield_across_state_future_ssp245_s = np.load(input_path_projection+'yield_across_state_future_ssp245_s.npy')
yield_across_state_future_ssp585 = np.load(input_path_projection+'yield_across_state_future_ssp585.npy')
yield_across_state_future_ssp585_s = np.load(input_path_projection+'yield_across_state_future_ssp585_s.npy')

yield_average_model_hist_ssp245 = np.load(input_path_projection+'yield_average_model_hist_ssp245.npy')
yield_average_model_hist_ssp245_s = np.load(input_path_projection+'yield_average_model_hist_ssp245_s.npy')
yield_average_model_hist_ssp585 = np.load(input_path_projection+'yield_average_model_hist_ssp585.npy')
yield_average_model_hist_ssp585_s = np.load(input_path_projection+'yield_average_model_hist_ssp585_s.npy')
yield_average_model_future_ssp245 = np.load(input_path_projection+'yield_average_model_future_ssp245.npy')
yield_average_model_future_ssp245_s = np.load(input_path_projection+'yield_average_model_future_ssp245_s.npy')
yield_average_model_future_ssp585 = np.load(input_path_projection+'yield_average_model_future_ssp585.npy')
yield_average_model_future_ssp585_s = np.load(input_path_projection+'yield_average_model_future_ssp585_s.npy')

yield_all_hist_ssp245 = np.load(input_path_projection+'yield_all_hist_ssp245.npy')
yield_all_hist_ssp245_s = np.load(input_path_projection+'yield_all_hist_ssp245_s.npy')
yield_all_hist_ssp585 = np.load(input_path_projection+'yield_all_hist_ssp585.npy')
yield_all_hist_ssp585_s = np.load(input_path_projection+'yield_all_hist_ssp585_s.npy')

yield_all_model_hist_ssp245 = np.load(input_path_projection+'yield_all_model_hist_ssp245.npy')
yield_all_model_hist_ssp245_s = np.load(input_path_projection+'yield_all_model_hist_ssp245_s.npy')
yield_all_model_future_ssp245 = np.load(input_path_projection+'yield_all_model_future_ssp245.npy')
yield_all_model_future_ssp245_s = np.load(input_path_projection+'yield_all_model_future_ssp245_s.npy')

yield_all_model_hist_ssp585 = np.load(input_path_projection+'yield_all_model_hist_ssp585.npy')
yield_all_model_hist_ssp585_s = np.load(input_path_projection+'yield_all_model_hist_ssp585_s.npy')
yield_all_model_future_ssp585 = np.load(input_path_projection+'yield_all_model_future_ssp585.npy')
yield_all_model_future_ssp585_s = np.load(input_path_projection+'yield_all_model_future_ssp585_s.npy')

yield_all_future_rcp45 = np.load(input_path_projection_MACA+'yield_all_future_rcp45.npy')
yield_all_future_rcp45_s = np.load(input_path_projection_MACA+'yield_all_future_rcp45_s.npy')
yield_all_future_rcp45_m = np.load(input_path_projection_MACA+'yield_all_future_rcp45_m.npy')
yield_all_future_rcp85 = np.load(input_path_projection_MACA+'yield_all_future_rcp85.npy')
yield_all_future_rcp85_s = np.load(input_path_projection_MACA+'yield_all_future_rcp85_s.npy')
yield_all_future_rcp85_m = np.load(input_path_projection_MACA+'yield_all_future_rcp85_m.npy')
yield_all_hist_rcp45 = np.load(input_path_projection_MACA+'yield_all_hist_rcp45.npy')
yield_all_hist_rcp45_s = np.load(input_path_projection_MACA+'yield_all_hist_rcp45_s.npy')
yield_all_hist_rcp45_m = np.load(input_path_projection_MACA+'yield_all_hist_rcp45_m.npy')
yield_all_hist_rcp85 = np.load(input_path_projection_MACA+'yield_all_hist_rcp85.npy')
yield_all_hist_rcp85_s = np.load(input_path_projection_MACA+'yield_all_hist_rcp85_s.npy')
yield_all_hist_rcp85_m = np.load(input_path_projection_MACA+'yield_all_hist_rcp85_m.npy')

yield_across_state_hist_rcp45 = np.load(input_path_projection_MACA+'yield_across_state_hist_rcp45.npy')
yield_across_state_hist_rcp45_s = np.load(input_path_projection_MACA+'yield_across_state_hist_rcp45_s.npy')
yield_across_state_hist_rcp85 = np.load(input_path_projection_MACA+'yield_across_state_hist_rcp85.npy')
yield_across_state_hist_rcp85_s = np.load(input_path_projection_MACA+'yield_across_state_hist_rcp85_s.npy')
yield_across_state_future_rcp45 = np.load(input_path_projection_MACA+'yield_across_state_future_rcp45.npy')
yield_across_state_future_rcp45_s = np.load(input_path_projection_MACA+'yield_across_state_future_rcp45_s.npy')
yield_across_state_future_rcp85 = np.load(input_path_projection_MACA+'yield_across_state_future_rcp85.npy')
yield_across_state_future_rcp85_s = np.load(input_path_projection_MACA+'yield_across_state_future_rcp85_s.npy')

yield_average_model_hist_rcp45 = np.load(input_path_projection_MACA+'yield_average_model_hist_rcp45.npy')
yield_average_model_hist_rcp45_s = np.load(input_path_projection_MACA+'yield_average_model_hist_rcp45_s.npy')
yield_average_model_hist_rcp85 = np.load(input_path_projection_MACA+'yield_average_model_hist_rcp85.npy')
yield_average_model_hist_rcp85_s = np.load(input_path_projection_MACA+'yield_average_model_hist_rcp85_s.npy')
yield_average_model_future_rcp45 = np.load(input_path_projection_MACA+'yield_average_model_future_rcp45.npy')
yield_average_model_future_rcp45_s = np.load(input_path_projection_MACA+'yield_average_model_future_rcp45_s.npy')
yield_average_model_future_rcp85 = np.load(input_path_projection_MACA+'yield_average_model_future_rcp85.npy')
yield_average_model_future_rcp85_s = np.load(input_path_projection_MACA+'yield_average_model_future_rcp85_s.npy')

yield_all_hist_rcp45 = np.load(input_path_projection_MACA+'yield_all_hist_rcp45.npy')
yield_all_hist_rcp45_s = np.load(input_path_projection_MACA+'yield_all_hist_rcp45_s.npy')
yield_all_hist_rcp85 = np.load(input_path_projection_MACA+'yield_all_hist_rcp85.npy')
yield_all_hist_rcp85_s = np.load(input_path_projection_MACA+'yield_all_hist_rcp85_s.npy')

yield_all_model_hist_rcp45 = np.load(input_path_projection_MACA+'yield_all_model_hist_rcp45.npy')
yield_all_model_hist_rcp45_s = np.load(input_path_projection_MACA+'yield_all_model_hist_rcp45_s.npy')
yield_all_model_future_rcp45 = np.load(input_path_projection_MACA+'yield_all_model_future_rcp45.npy')
yield_all_model_future_rcp45_s = np.load(input_path_projection_MACA+'yield_all_model_future_rcp45_s.npy')

yield_all_model_hist_rcp85 = np.load(input_path_projection_MACA+'yield_all_model_hist_rcp85.npy')
yield_all_model_hist_rcp85_s = np.load(input_path_projection_MACA+'yield_all_model_hist_rcp85_s.npy')
yield_all_model_future_rcp85 = np.load(input_path_projection_MACA+'yield_all_model_future_rcp85.npy')
yield_all_model_future_rcp85_s = np.load(input_path_projection_MACA+'yield_all_model_future_rcp85_s.npy')

ACI_list = ['Dormancy_Chill','Dormancy_ETo','Jan_P','Bloom_P','Bloom_Tmin' ,'Bloom_FrostDays','Bloom_ETo', 'Bloom_GDD','Bloom_Humidity','Bloom_WindyDays','Growing_ETo','Growing_GDD', 'Growing_KDD','Harvest_P']

aci_contribution_ssp245_total = np.load(input_path_contribution+'aci_contribution_ssp245_total.npy')
aci_contribution_ssp585_total = np.load(input_path_contribution+'aci_contribution_ssp585_total.npy')
aci_contribution_ssp245_county_total = np.load(input_path_contribution+'aci_contribution_ssp245_county_total.npy')
aci_contribution_ssp585_county_total = np.load(input_path_contribution+'aci_contribution_ssp585_county_total.npy')

aci_contribution_ssp245_2001_2020 = np.mean(aci_contribution_ssp245_total[21:41,:,:], axis=0)
aci_contribution_ssp585_2001_2020 = np.mean(aci_contribution_ssp585_total[21:41,:,:], axis=0)

aci_contribution_ssp245_2041_2060 = np.mean(aci_contribution_ssp245_total[60:80,:,:], axis=0)
aci_contribution_ssp585_2041_2060 = np.mean(aci_contribution_ssp585_total[60:80,:,:], axis=0)

aci_contribution_ssp245_2080_2099 = np.mean(aci_contribution_ssp245_total[100:120,:,:], axis=0)
aci_contribution_ssp585_2080_2099 = np.mean(aci_contribution_ssp585_total[100:120,:,:], axis=0)

aci_contribution_ssp245_change_percent_2041_2060 = np.zeros((1000,aci_num))
aci_contribution_ssp585_change_percent_2041_2060 = np.zeros((1000,aci_num))
aci_contribution_ssp245_change_percent_2080_2099 = np.zeros((1000,aci_num))
aci_contribution_ssp585_change_percent_2080_2099 = np.zeros((1000,aci_num))
aci_contribution_ssp245_change_2041_2060 = np.zeros((1000,aci_num))
aci_contribution_ssp585_change_2041_2060 = np.zeros((1000,aci_num))
aci_contribution_ssp245_change_2080_2099 = np.zeros((1000,aci_num))
aci_contribution_ssp585_change_2080_2099 = np.zeros((1000,aci_num))

aci_contribution_ssp245_change_total_2041_2060 = np.zeros((1000))
aci_contribution_ssp585_change_total_2041_2060 = np.zeros((1000))
aci_contribution_ssp245_change_total_2080_2099 = np.zeros((1000))
aci_contribution_ssp585_change_total_2080_2099 = np.zeros((1000))
for i in range(0,1000):
    aci_contribution_ssp245_change_total_2041_2060[i] = np.sum(aci_contribution_ssp245_2041_2060[i,:])-np.sum(aci_contribution_ssp245_2001_2020[i,:])
    aci_contribution_ssp585_change_total_2041_2060[i] = np.sum(aci_contribution_ssp585_2041_2060[i,:])-np.sum(aci_contribution_ssp585_2001_2020[i,:])
    aci_contribution_ssp245_change_total_2080_2099[i] = np.sum(aci_contribution_ssp245_2080_2099[i,:])-np.sum(aci_contribution_ssp245_2001_2020[i,:])
    aci_contribution_ssp585_change_total_2080_2099[i] = np.sum(aci_contribution_ssp585_2080_2099[i,:])-np.sum(aci_contribution_ssp585_2001_2020[i,:])
    aci_contribution_ssp245_change_percent_2041_2060[i,:] = 100*(aci_contribution_ssp245_2041_2060[i,:]-aci_contribution_ssp245_2001_2020[i,:])/np.absolute(aci_contribution_ssp245_change_total_2041_2060[i])
    aci_contribution_ssp585_change_percent_2041_2060[i,:] = 100*(aci_contribution_ssp585_2041_2060[i,:]-aci_contribution_ssp585_2001_2020[i,:])/np.absolute(aci_contribution_ssp585_change_total_2041_2060[i])
    aci_contribution_ssp245_change_percent_2080_2099[i,:] = 100*(aci_contribution_ssp245_2080_2099[i,:]-aci_contribution_ssp245_2001_2020[i,:])/np.absolute(aci_contribution_ssp245_change_total_2080_2099[i])
    aci_contribution_ssp585_change_percent_2080_2099[i,:] = 100*(aci_contribution_ssp585_2080_2099[i,:]-aci_contribution_ssp585_2001_2020[i,:])/np.absolute(aci_contribution_ssp585_change_total_2080_2099[i])
    aci_contribution_ssp245_change_2041_2060[i,:] = (aci_contribution_ssp245_2041_2060[i,:]-aci_contribution_ssp245_2001_2020[i,:])
    aci_contribution_ssp585_change_2041_2060[i,:] = (aci_contribution_ssp585_2041_2060[i,:]-aci_contribution_ssp585_2001_2020[i,:])
    aci_contribution_ssp245_change_2080_2099[i,:] = (aci_contribution_ssp245_2080_2099[i,:]-aci_contribution_ssp245_2001_2020[i,:])
    aci_contribution_ssp585_change_2080_2099[i,:] = (aci_contribution_ssp585_2080_2099[i,:]-aci_contribution_ssp585_2001_2020[i,:])


median_ssp245_2041_2060 = np.nanmedian(aci_contribution_ssp245_change_percent_2041_2060, axis=0)
median_ssp585_2041_2060 = np.nanmedian(aci_contribution_ssp585_change_percent_2041_2060, axis=0)
median_ssp245_2080_2099 = np.nanmedian(aci_contribution_ssp245_change_percent_2080_2099, axis=0)
median_ssp585_2080_2099 = np.nanmedian(aci_contribution_ssp585_change_percent_2080_2099, axis=0)

aci_contribution_ssp245_county_2050_change_percent = np.load(input_path_contribution+'aci_contribution_ssp245_county_2050_change_percent.npy')
aci_contribution_ssp245_county_2090_change_percent = np.load(input_path_contribution+'aci_contribution_ssp245_county_2090_change_percent.npy')
aci_contribution_ssp585_county_2050_change_percent = np.load(input_path_contribution+'aci_contribution_ssp585_county_2050_change_percent.npy')
aci_contribution_ssp585_county_2090_change_percent = np.load(input_path_contribution+'aci_contribution_ssp585_county_2090_change_percent.npy')

aci_contribution_ssp245_county_2050_change_percent_median = np.median(aci_contribution_ssp245_county_2050_change_percent, axis=1)
aci_contribution_ssp245_county_2090_change_percent_median = np.median(aci_contribution_ssp245_county_2090_change_percent, axis=1)
aci_contribution_ssp585_county_2050_change_percent_median = np.median(aci_contribution_ssp585_county_2050_change_percent, axis=1)
aci_contribution_ssp585_county_2090_change_percent_median = np.median(aci_contribution_ssp585_county_2090_change_percent, axis=1)

aci_contribution_rcp45_total = np.load(input_path_contribution_MACA+'aci_contribution_rcp45_total.npy')
aci_contribution_rcp85_total = np.load(input_path_contribution_MACA+'aci_contribution_rcp85_total.npy')
aci_contribution_rcp45_county_total = np.load(input_path_contribution_MACA+'aci_contribution_rcp45_county_total.npy')
aci_contribution_rcp85_county_total = np.load(input_path_contribution_MACA+'aci_contribution_rcp85_county_total.npy')

aci_contribution_rcp45_2001_2020 = np.mean(aci_contribution_rcp45_total[21:41,:,:], axis=0)
aci_contribution_rcp85_2001_2020 = np.mean(aci_contribution_rcp85_total[21:41,:,:], axis=0)

aci_contribution_rcp45_2041_2060 = np.mean(aci_contribution_rcp45_total[60:80,:,:], axis=0)
aci_contribution_rcp85_2041_2060 = np.mean(aci_contribution_rcp85_total[60:80,:,:], axis=0)

aci_contribution_rcp45_2080_2099 = np.mean(aci_contribution_rcp45_total[100:120,:,:], axis=0)
aci_contribution_rcp85_2080_2099 = np.mean(aci_contribution_rcp85_total[100:120,:,:], axis=0)

aci_contribution_rcp45_change_percent_2041_2060 = np.zeros((1000,aci_num))
aci_contribution_rcp85_change_percent_2041_2060 = np.zeros((1000,aci_num))
aci_contribution_rcp45_change_percent_2080_2099 = np.zeros((1000,aci_num))
aci_contribution_rcp85_change_percent_2080_2099 = np.zeros((1000,aci_num))
aci_contribution_rcp45_change_2041_2060 = np.zeros((1000,aci_num))
aci_contribution_rcp85_change_2041_2060 = np.zeros((1000,aci_num))
aci_contribution_rcp45_change_2080_2099 = np.zeros((1000,aci_num))
aci_contribution_rcp85_change_2080_2099 = np.zeros((1000,aci_num))

aci_contribution_rcp45_change_total_2041_2060 = np.zeros((1000))
aci_contribution_rcp85_change_total_2041_2060 = np.zeros((1000))
aci_contribution_rcp45_change_total_2080_2099 = np.zeros((1000))
aci_contribution_rcp85_change_total_2080_2099 = np.zeros((1000))
for i in range(0,1000):
    aci_contribution_rcp45_change_total_2041_2060[i] = np.sum(aci_contribution_rcp45_2041_2060[i,:])-np.sum(aci_contribution_rcp45_2001_2020[i,:])
    aci_contribution_rcp85_change_total_2041_2060[i] = np.sum(aci_contribution_rcp85_2041_2060[i,:])-np.sum(aci_contribution_rcp85_2001_2020[i,:])
    aci_contribution_rcp45_change_total_2080_2099[i] = np.sum(aci_contribution_rcp45_2080_2099[i,:])-np.sum(aci_contribution_rcp45_2001_2020[i,:])
    aci_contribution_rcp85_change_total_2080_2099[i] = np.sum(aci_contribution_rcp85_2080_2099[i,:])-np.sum(aci_contribution_rcp85_2001_2020[i,:])
    aci_contribution_rcp45_change_percent_2041_2060[i,:] = 100*(aci_contribution_rcp45_2041_2060[i,:]-aci_contribution_rcp45_2001_2020[i,:])/np.absolute(aci_contribution_rcp45_change_total_2041_2060[i])
    aci_contribution_rcp85_change_percent_2041_2060[i,:] = 100*(aci_contribution_rcp85_2041_2060[i,:]-aci_contribution_rcp85_2001_2020[i,:])/np.absolute(aci_contribution_rcp85_change_total_2041_2060[i])
    aci_contribution_rcp45_change_percent_2080_2099[i,:] = 100*(aci_contribution_rcp45_2080_2099[i,:]-aci_contribution_rcp45_2001_2020[i,:])/np.absolute(aci_contribution_rcp45_change_total_2080_2099[i])
    aci_contribution_rcp85_change_percent_2080_2099[i,:] = 100*(aci_contribution_rcp85_2080_2099[i,:]-aci_contribution_rcp85_2001_2020[i,:])/np.absolute(aci_contribution_rcp85_change_total_2080_2099[i])
    aci_contribution_rcp45_change_2041_2060[i,:] = (aci_contribution_rcp45_2041_2060[i,:]-aci_contribution_rcp45_2001_2020[i,:])
    aci_contribution_rcp85_change_2041_2060[i,:] = (aci_contribution_rcp85_2041_2060[i,:]-aci_contribution_rcp85_2001_2020[i,:])
    aci_contribution_rcp45_change_2080_2099[i,:] = (aci_contribution_rcp45_2080_2099[i,:]-aci_contribution_rcp45_2001_2020[i,:])
    aci_contribution_rcp85_change_2080_2099[i,:] = (aci_contribution_rcp85_2080_2099[i,:]-aci_contribution_rcp85_2001_2020[i,:])


median_rcp45_2041_2060 = np.nanmedian(aci_contribution_rcp45_change_percent_2041_2060, axis=0)
median_rcp85_2041_2060 = np.nanmedian(aci_contribution_rcp85_change_percent_2041_2060, axis=0)
median_rcp45_2080_2099 = np.nanmedian(aci_contribution_rcp45_change_percent_2080_2099, axis=0)
median_rcp85_2080_2099 = np.nanmedian(aci_contribution_rcp85_change_percent_2080_2099, axis=0)

aci_contribution_rcp45_county_2050_change_percent = np.load(input_path_contribution+'aci_contribution_rcp45_county_2050_change_percent.npy')
aci_contribution_rcp45_county_2090_change_percent = np.load(input_path_contribution+'aci_contribution_rcp45_county_2090_change_percent.npy')
aci_contribution_rcp85_county_2050_change_percent = np.load(input_path_contribution+'aci_contribution_rcp85_county_2050_change_percent.npy')
aci_contribution_rcp85_county_2090_change_percent = np.load(input_path_contribution+'aci_contribution_rcp85_county_2090_change_percent.npy')

aci_contribution_rcp45_county_2050_change_percent_median = np.median(aci_contribution_rcp45_county_2050_change_percent, axis=1)
aci_contribution_rcp45_county_2090_change_percent_median = np.median(aci_contribution_rcp45_county_2090_change_percent, axis=1)
aci_contribution_rcp85_county_2050_change_percent_median = np.median(aci_contribution_rcp85_county_2050_change_percent, axis=1)
aci_contribution_rcp85_county_2090_change_percent_median = np.median(aci_contribution_rcp85_county_2090_change_percent, axis=1)




##load coefficient from lasso 1000 models 
aci_num = 14
for i in range(1,11):
    locals()['coef'+str(i)] = np.zeros((100,aci_num*2+32))
coef_sum = np.zeros((0,aci_num*2+32))
for i in range(1,1001):
    locals()['coef'+str(((i-1)//100)+1)][i%100-1] = genfromtxt(input_path_model+'coef_'+str(i)+'.csv', delimiter = ',')
for i in range(1,11):
    coef_sum = np.row_stack((coef_sum, locals()['coef'+str(i)]))
area = genfromtxt(input_path+'almond_area.csv', delimiter = ',')
production = genfromtxt(input_path+'almond_production.csv', delimiter = ',')
gridmet = genfromtxt(input_path_gridmet+'Gridmet_std_11_9_Frost_Chill_35.csv', delimiter = ',')
yield_csv = genfromtxt(input_path+'almond_yield_1980_2020.csv', delimiter = ',')

simulation_gridmet = np.zeros((656, 1000))
production_gridmet = np.zeros((656, 1000))
for trial in range(1,11):
    for i in range(0,656):
        for j in range(0,100):
            simulation_gridmet[i,j+((trial-1)*100)] = np.nansum(gridmet[i,:]*locals()['coef'+str(trial)][j,:])
for index in range(0,16):
    for year in range(0,41):
        production_gridmet[index*41+year,:] = simulation_gridmet[index*41+year,:]*area[year,index%16]
production_gridmet_split = np.split(production_gridmet,16) 
production_gridmet_all = np.zeros((41,1000)) 
for county_id in range(0,16):
    for year in range(0,41):
        production_gridmet_all[year,:] = production_gridmet_all[year,:]+production_gridmet_split[county_id][year,:]
yield_gridmet_state = np.zeros((41,1000))
for year in range(0,41):
    yield_gridmet_state[year,:] = production_gridmet_all[year,:]/np.sum(area[year])

production_observed_all = np.sum(production, axis = 1)
yield_observed_state = np.zeros((41))
for year in range(0,41):
    yield_observed_state[year] = production_observed_all[year]/np.sum(area[year])


tech_trend_county_con = np.zeros((16,120,1000))
tech_trend_county_int = np.zeros((16,120,1000))
for county_id in range(0,16):
    for trial in range(1000):
        tech_trend_county_con[county_id, :, trial] = coef_sum[:,county_id + aci_num*2][trial]
        tech_trend_county_int[county_id, :, trial] = coef_sum[:,county_id + aci_num*2][trial]
tech_year_var_con = np.arange(1,121)
tech_year_var_int = np.zeros(120)
tech_year_var_int[0:40] = np.arange(1,41)
tech_year_var_int[40] = 41
for i in range(1,80):
    tech_year_var_int[i+40] = tech_year_var_int[i+40-1] + (80 - i)/80
for year in range(1,121):
    tech_trend_county_con[:, year-1,:] = tech_trend_county_con[:, year-1,:] * tech_year_var_con[year-1]
    tech_trend_county_int[:, year-1,:] = tech_trend_county_int[:, year-1,:] * tech_year_var_int[year-1]

future_tech_trend_county_ssp245_con = np.zeros((120,16,1000))
future_tech_trend_county_ssp245_int = np.zeros((120,16,1000))
for i in range(16):
    for year in range(0,120):
        for trial in range(1000):
            future_tech_trend_county_ssp245_con[year,i,trial] = tech_trend_county_con[i,year,trial] + np.mean(np.split(yield_all_model_hist_ssp245_s,16)[i][-1,:].reshape(8,1000), axis=0)[trial]
            future_tech_trend_county_ssp245_int[year,i,trial] = tech_trend_county_int[i,year,trial] + np.mean(np.split(yield_all_model_hist_ssp245_s,16)[i][-1,:].reshape(8,1000), axis=0)[trial]

future_tech_trend_county_rcp45_con = np.zeros((120,16,1000))
future_tech_trend_county_rcp45_int = np.zeros((120,16,1000))
for i in range(16):
    for year in range(0,120):
        for trial in range(1000):
            future_tech_trend_county_rcp45_con[year,i,trial] = tech_trend_county_con[i,year,trial] + np.mean(np.split(yield_all_model_hist_rcp45_s,16)[i][-1,:].reshape(18,1000), axis=0)[trial]
            future_tech_trend_county_rcp45_int[year,i,trial] = tech_trend_county_int[i,year,trial] + np.mean(np.split(yield_all_model_hist_rcp45_s,16)[i][-1,:].reshape(18,1000), axis=0)[trial]


##calculate reference average historical yield
model_list = ['ACCESS-CM2', 'EC-Earth3', 'EC-Earth3-Veg', 'GFDL-ESM4',  'INM-CM5-0',  'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'CNRM-ESM2-1']
ACI_list = ['Dormancy_Chill','Dormancy_ETo','Jan_P','Bloom_P','Bloom_Tmin' ,'Bloom_FrostDays','Bloom_ETo', 'Bloom_GDD','Bloom_Humidity','Bloom_WindyDays','Growing_ETo','Growing_GDD', 'Growing_KDD','Harvest_P']
historical_years = 20
aci_ssp245_hist_sum = np.zeros((8,16,historical_years,aci_num*2+32))
aci_ssp585_hist_sum = np.zeros((8,16,historical_years,aci_num*2+32))

for model in range(8):
    for county in range(16):
        aci_ssp245 = np.split(genfromtxt(input_path_LOCA+str(model_list[model])+'hist_ssp245_ACI.csv', delimiter = ','),16)[county][-20:]
        aci_ssp585 = np.split(genfromtxt(input_path_LOCA+str(model_list[model])+'hist_ssp585_ACI.csv', delimiter = ','),16)[county][-20:]
        aci_ssp245[:,aci_num*2+county] = 41
        aci_ssp585[:,aci_num*2+county] = 41
        aci_ssp245_hist_sum[model,county,:,:] = aci_ssp245
        aci_ssp585_hist_sum[model,county,:,:] = aci_ssp585
reference_yield_2001_2020_ssp245 = np.zeros((8,1000,16,historical_years))
reference_yield_2001_2020_ssp585 = np.zeros((8,1000,16,historical_years))
for model in range(8):
    for trial in range(1000):
        for county in range(16):
            for year in range(historical_years):
                reference_yield_2001_2020_ssp245[model,trial,county,year] = np.sum(aci_ssp245_hist_sum[model, county,year,:]*coef_sum[trial,:])
                reference_yield_2001_2020_ssp585[model,trial,county,year] = np.sum(aci_ssp585_hist_sum[model, county,year,:]*coef_sum[trial,:])

reference_yield_2001_2020_ssp245_state = np.zeros((8,1000,historical_years))
reference_yield_2001_2020_ssp585_state = np.zeros((8,1000,historical_years))

for year in range(historical_years):
    for county in range(16):
        reference_yield_2001_2020_ssp245_state[:,:,year] = reference_yield_2001_2020_ssp245_state[:,:,year] + reference_yield_2001_2020_ssp245[:,:,county,year] * area[-20:,county][year] / np.sum(area[-20:,:][year])
        reference_yield_2001_2020_ssp585_state[:,:,year] = reference_yield_2001_2020_ssp585_state[:,:,year] + reference_yield_2001_2020_ssp585[:,:,county,year] * area[-20:,county][year] / np.sum(area[-20:,:][year])
reference_yield_2001_2020_ssp245_state_median = np.mean(np.median(np.mean(reference_yield_2001_2020_ssp245_state,axis=0),axis=0))
reference_yield_2001_2020_ssp585_state_median = np.mean(np.median(np.mean(reference_yield_2001_2020_ssp585_state,axis=0),axis=0))
reference_yield_2001_2020_ssp245_state_flatten = np.ndarray.flatten(np.mean(reference_yield_2001_2020_ssp245_state,axis=2))
reference_yield_2001_2020_ssp585_state_flatten = np.ndarray.flatten(np.mean(reference_yield_2001_2020_ssp585_state,axis=2))


model_list_MACA = ['bcc-csm1-1-m', 'bcc-csm1-1','BNU-ESM', 'CanESM2', 'CSIRO-Mk3-6-0', 'GFDL-ESM2G', 'GFDL-ESM2M', 'inmcm4', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR','CNRM-CM5', 'HadGEM2-CC365','HadGEM2-ES365', 'IPSL-CM5B-LR', 'MIROC5', 'MIROC-ESM', 'MIROC-ESM-CHEM','MRI-CGCM3']
historical_years = 20
aci_rcp45_hist_sum = np.zeros((18,16,historical_years,aci_num*2+32))
aci_rcp85_hist_sum = np.zeros((18,16,historical_years,aci_num*2+32))

for model in range(18):
    for county in range(16):
        aci_rcp45 = np.split(genfromtxt(input_path_MACA+str(model_list_MACA[model])+'hist_rcp45_ACI.csv', delimiter = ','),16)[county][-20:]
        aci_rcp85 = np.split(genfromtxt(input_path_MACA+str(model_list_MACA[model])+'hist_rcp85_ACI.csv', delimiter = ','),16)[county][-20:]
        aci_rcp45[:,aci_num*2+county] = 41
        aci_rcp85[:,aci_num*2+county] = 41
        aci_rcp45_hist_sum[model,county,:,:] = aci_rcp45
        aci_rcp85_hist_sum[model,county,:,:] = aci_rcp85
reference_yield_2001_2020_rcp45 = np.zeros((18,1000,16,historical_years))
reference_yield_2001_2020_rcp85 = np.zeros((18,1000,16,historical_years))
for model in range(18):
    for trial in range(1000):
        for county in range(16):
            for year in range(historical_years):
                reference_yield_2001_2020_rcp45[model,trial,county,year] = np.sum(aci_rcp45_hist_sum[model, county,year,:]*coef_sum[trial,:])
                reference_yield_2001_2020_rcp85[model,trial,county,year] = np.sum(aci_rcp85_hist_sum[model, county,year,:]*coef_sum[trial,:])

reference_yield_2001_2020_rcp45_state = np.zeros((18,1000,historical_years))
reference_yield_2001_2020_rcp85_state = np.zeros((18,1000,historical_years))

for year in range(historical_years):
    for county in range(16):
        reference_yield_2001_2020_rcp45_state[:,:,year] = reference_yield_2001_2020_rcp45_state[:,:,year] + reference_yield_2001_2020_rcp45[:,:,county,year] * area[-20:,county][year] / np.sum(area[-20:,:][year])
        reference_yield_2001_2020_rcp85_state[:,:,year] = reference_yield_2001_2020_rcp85_state[:,:,year] + reference_yield_2001_2020_rcp85[:,:,county,year] * area[-20:,county][year] / np.sum(area[-20:,:][year])
reference_yield_2001_2020_rcp45_state_median = np.mean(np.median(np.mean(reference_yield_2001_2020_rcp45_state,axis=0),axis=0))
reference_yield_2001_2020_rcp85_state_median = np.mean(np.median(np.mean(reference_yield_2001_2020_rcp85_state,axis=0),axis=0))
reference_yield_2001_2020_rcp45_state_flatten = np.ndarray.flatten(np.mean(reference_yield_2001_2020_rcp45_state,axis=2))
reference_yield_2001_2020_rcp85_state_flatten = np.ndarray.flatten(np.mean(reference_yield_2001_2020_rcp85_state,axis=2))



##Figure 1:Evaluation for LOCA
R2_train_sum = np.zeros((1000))
for i in range(1,1001):
    R2_train_sum[i-1] = genfromtxt(str(input_path_model)+'/score_train_'+str(i)+'.csv', delimiter = ',')

R2_test_sum = np.zeros((1000))
for i in range(1,1001):
    R2_test_sum[i-1] = genfromtxt(str(input_path_model)+'/score_test_'+str(i)+'.csv', delimiter = ',')
R2 = np.column_stack((R2_train_sum, R2_test_sum))
labels = ['Training set', 'Testing set']
colors = ['yellow', 'goldenrod']
fig = plt.figure(figsize = (35,12))
gs = gridspec.GridSpec(1, 100)
#plt.suptitle('Statiscal model (Lasso) performance', fontsize = 35, y = 1.02, x = 0.5)
ax0 = fig.add_subplot(gs[0, 0:16])
box = ax0.boxplot(R2, patch_artist= True, labels = labels, boxprops={'linewidth' : 2}, whiskerprops={'linewidth' : 3},capprops={'linewidth' : 3}, medianprops={'color' : 'black', 'linewidth' : 3}, widths = 0.7,showfliers=False)
ax0.set_ylabel(r'$R^2$', fontsize=35)
ax0.tick_params(axis='y', which='major', labelsize=35)
ax0.tick_params(axis='x', which='major', labelsize=35, rotation = 60)
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set(linewidth = 2)
plt.text(0, 0.873, 'a', fontsize = 35, fontweight='bold')
ax1 = fig.add_subplot(gs[0, 23:58])
ax1.plot(np.arange(1980,2021,1),yield_observed_state[0:41], label = 'Actual Yield',linewidth = 4, color = 'green')
t = np.arange(1980,2021)
tsplot(t,np.transpose(yield_gridmet_state[0:41]), color = 'darkorange')
plt.plot(np.arange(1980,2021,1),np.median(yield_gridmet_state[0:41],axis=1), label = 'GridMet-modeled median', color = 'darkorange', linestyle = 'solid')
plt.xticks(np.arange(1980,2021,4),np.arange(1980,2021,4), fontsize = 35, rotation = 60)
plt.yticks(fontsize = 35)
plt.ylim(0,1.6)
plt.xlim(1980,2021)
plt.ylabel('Yield ton/acre', fontsize = 35)
darkorange_patch = mpatches.Patch(color = 'darkorange',label = 'GridMet')
plt.legend(handles = [darkorange_patch, Line2D([0], [0], color='green', lw=4, label='Observed')],loc = 'upper left', fontsize = 35)
#plt.title('CA area-weighted almond yield', fontsize = 35, x = 1)
box_legend = np.zeros((1000,3))
box_legend[:,0] = np.random.normal(1.45,0.06,size = (1000))
box_legend[:,1] = box_legend[:,0]
box_legend[:,2] = box_legend[:,0]
tsplot(np.arange(2000,2003), box_legend,color = 'darkorange')
plt.text(x = 2003,y = 1.31, s='95% CI', fontsize = 26)
plt.text(x = 2003,y = 1.37, s='67% CI', fontsize = 26)
plt.text(x = 2003,y = 1.43, s='Median', fontsize = 26)
plt.plot(np.arange(2000,2003),np.median(box_legend,axis=0), color = 'darkorange', linestyle = 'solid', linewidth = 4)
plt.text(1974, 1.6, 'b', fontsize = 35, fontweight='bold')
ax2 = fig.add_subplot(gs[0, 65:100])
ax2.plot(np.arange(1980,2021), np.median(yield_all_hist_ssp245, axis=1) , color = 'black', linewidth =4)
tsplot(np.arange(1980,2021), np.transpose(yield_all_hist_ssp245) , color = 'grey')
ax2.plot(np.arange(1980,2021,1),yield_observed_state[0:41], label = 'Actual Yield',linewidth = 4, color = 'green')
plt.xticks(np.arange(1980,2021,4),np.arange(1980,2021,4), fontsize = 35, rotation = 60)
plt.yticks(fontsize = 35)
plt.ylim(0,1.6)
grey_patch = mpatches.Patch(color = 'grey',label = 'SSP245')
plt.legend(handles = [grey_patch, Line2D([0], [0], color='green', lw=4, label='Observed')],loc = 'upper left', fontsize = 35)
box_legend = np.zeros((1000,3))
box_legend[:,0] = np.random.normal(1.45,0.06,size = (1000))
box_legend[:,1] = box_legend[:,0]
box_legend[:,2] = box_legend[:,0]
tsplot(np.arange(2004,2007), box_legend,color = 'grey')
plt.text(x = 2007,y = 1.31, s='95% CI', fontsize = 26)
plt.text(x = 2007,y = 1.37, s='67% CI', fontsize = 26)
plt.text(x = 2007,y = 1.43, s='Median', fontsize = 26)
plt.plot(np.arange(2004,2007),np.median(box_legend,axis=0), color = 'black', linestyle = 'solid', linewidth = 4)
plt.ylabel('Yield ton/acre', fontsize = 35)
plt.xlim(1980,2021)
plt.text(1974, 1.6, 'c', fontsize = 35, fontweight='bold')
plt.savefig(str(save_path)+'/evaluation.pdf',bbox_inches='tight', dpi = 300)


##Figure 2: yield time series for LOCA
df_2080_2099_20yrmean_yield_ssp245 = pd.DataFrame({'scenario' : np.repeat('ssp245', 8000),'mean_yield' : np.mean(yield_all_future_ssp245[-20:],axis=0), 'tech' : np.repeat('yes',8000)})
df_2080_2099_20yrmean_yield_ssp245_s = pd.DataFrame({'scenario' : np.repeat('ssp245', 8000),'mean_yield' : np.mean(yield_all_future_ssp245_s[-20:],axis=0), 'tech' : np.repeat('no',8000)})
df_2080_2099_20yrmean_yield_ssp245_m = pd.DataFrame({'scenario' : np.repeat('ssp245', 8000),'mean_yield' : np.mean(yield_all_future_ssp245_m[-20:],axis=0), 'tech' : np.repeat('m',8000)})
df_2080_2099_20yrmean_yield_ssp585 = pd.DataFrame({'scenario' : np.repeat('ssp585', 8000),'mean_yield' : np.mean(yield_all_future_ssp585[-20:],axis=0), 'tech' : np.repeat('yes',8000)})
df_2080_2099_20yrmean_yield_ssp585_s = pd.DataFrame({'scenario' : np.repeat('ssp585', 8000),'mean_yield' : np.mean(yield_all_future_ssp585_s[-20:],axis=0), 'tech' : np.repeat('no',8000)})
df_2080_2099_20yrmean_yield_ssp585_m = pd.DataFrame({'scenario' : np.repeat('ssp585', 8000),'mean_yield' : np.mean(yield_all_future_ssp585_m[-20:],axis=0), 'tech' : np.repeat('m',8000)})
df_2080_2099_20yrmean_yield_ssp245_total = pd.concat((df_2080_2099_20yrmean_yield_ssp245, df_2080_2099_20yrmean_yield_ssp245_s, df_2080_2099_20yrmean_yield_ssp245_m))
df_2080_2099_20yrmean_yield_ssp585_total = pd.concat((df_2080_2099_20yrmean_yield_ssp585, df_2080_2099_20yrmean_yield_ssp585_s, df_2080_2099_20yrmean_yield_ssp585_m))

yield_change_to_simulate2020_ssp245 = ((np.mean(yield_all_future_ssp245[-20:],axis=0) - reference_yield_2001_2020_ssp245_state_flatten)*100/reference_yield_2001_2020_ssp245_state_flatten).reshape(8,1000)
yield_change_to_simulate2020_ssp585 = ((np.mean(yield_all_future_ssp585[-20:],axis=0) - reference_yield_2001_2020_ssp585_state_flatten)*100/reference_yield_2001_2020_ssp585_state_flatten).reshape(8,1000)
yield_change_to_simulate2020_ssp245_s = ((np.mean(yield_all_future_ssp245_s[-20:],axis=0) - reference_yield_2001_2020_ssp245_state_flatten)*100/reference_yield_2001_2020_ssp245_state_flatten).reshape(8,1000)
yield_change_to_simulate2020_ssp585_s = ((np.mean(yield_all_future_ssp585_s[-20:],axis=0) - reference_yield_2001_2020_ssp585_state_flatten)*100/reference_yield_2001_2020_ssp585_state_flatten).reshape(8,1000)
yield_change_to_simulate2020_ssp245_m = ((np.mean(yield_all_future_ssp245_m[-20:],axis=0) - reference_yield_2001_2020_ssp245_state_flatten)*100/reference_yield_2001_2020_ssp245_state_flatten).reshape(8,1000)
yield_change_to_simulate2020_ssp585_m = ((np.mean(yield_all_future_ssp585_m[-20:],axis=0) - reference_yield_2001_2020_ssp585_state_flatten)*100/reference_yield_2001_2020_ssp585_state_flatten).reshape(8,1000)

yield_change_to_simulate2020_ssp245_ave = np.median(np.mean(yield_change_to_simulate2020_ssp245, axis=0), axis = 0)
yield_change_to_simulate2020_ssp585_ave = np.median(np.mean(yield_change_to_simulate2020_ssp585, axis=0), axis = 0)
yield_change_to_simulate2020_ssp245_s_ave = np.median(np.mean(yield_change_to_simulate2020_ssp245_s, axis=0), axis = 0)
yield_change_to_simulate2020_ssp585_s_ave = np.median(np.mean(yield_change_to_simulate2020_ssp585_s, axis=0), axis = 0)
yield_change_to_simulate2020_ssp245_m_ave = np.median(np.mean(yield_change_to_simulate2020_ssp245_m, axis=0), axis = 0)
yield_change_to_simulate2020_ssp585_m_ave = np.median(np.mean(yield_change_to_simulate2020_ssp585_m, axis=0), axis = 0)

##calculate num of year with % loss 
def get_num_year_with_loss(obs, pred):
    loss_percent_axis = np.linspace(0, 1, 21)
    matrix = np.zeros(len(loss_percent_axis))
    for i in range(len(loss_percent_axis)):
        matrix[i] = np.count_nonzero(pred<=(1-loss_percent_axis[i])*obs)
    return(matrix)

def extract_prob_curve(matrix, prob):
    min_value = np.min(np.abs(matrix-prob),axis=1)
    min_value_index = np.argmin(np.abs(matrix-prob),axis=1)
    for i in range(20):
        if min_value_index[i] > min_value_index[i+1]:
           min_value_index[i+1] = min_value_index[i]
    df = pd.DataFrame({'x' : np.arange(0,21), 'y' : min_value, 'min_index' : min_value_index})
    df = df.drop(index = np.where(min_value_index == np.min(min_value_index))[0][0:-1])
    df = df.drop(index = np.where(min_value_index == np.max(min_value_index))[0][1:])
    df = df.loc[df['y'] <=0.15]
    return(np.array(df.x), np.array(df.min_index))
    
def add_median_labels(ax, fmt='.1f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y+0.15, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='bold', color='white',fontsize = 25)
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])



def tsplot(x, y, n=2, percentile_min=2.5, percentile_max=97.5, color='r', plot_mean=False, plot_median=False, line_color='k', **kwargs):
    # calculate the lower and upper percentile groups, skipping 50 percentile
    perc1 = np.percentile(y, np.linspace(percentile_min, 31, num=n, endpoint=False), axis=0)
    perc2 = np.percentile(y, np.linspace(71.5, percentile_max, num=n+1)[1:], axis=0)

    if 'alpha' in kwargs:
        alpha = kwargs.pop('alpha')
    else:
        alpha = 1/n
    # fill lower and upper percentile groups
    for p1, p2 in zip(perc1, perc2):
        plt.fill_between(x, p1, p2, alpha=alpha, color=color, edgecolor='k')

    if plot_mean:
        plt.plot(x, np.mean(y, axis=0), color=line_color)


    if plot_median:
        plt.plot(x, np.median(y, axis=0), color=line_color)
    
    return plt.gca() 

def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])



                        

matrix_percent_loss_num_year_ssp245_2090 = np.zeros((21,8000))
matrix_percent_loss_num_year_ssp585_2090 = np.zeros((21,8000))
matrix_percent_loss_num_year_ssp245_2050 = np.zeros((21,8000))
matrix_percent_loss_num_year_ssp585_2050 = np.zeros((21,8000))
for trial in range(8000):
    matrix_percent_loss_num_year_ssp245_2090[:,trial] = get_num_year_with_loss(reference_yield_2001_2020_ssp245_state_flatten[trial], yield_all_future_ssp245_s[-20:,trial])
    matrix_percent_loss_num_year_ssp585_2090[:,trial] = get_num_year_with_loss(reference_yield_2001_2020_ssp585_state_flatten[trial], yield_all_future_ssp585_s[-20:,trial])
    matrix_percent_loss_num_year_ssp245_2050[:,trial] = get_num_year_with_loss(reference_yield_2001_2020_ssp245_state_flatten[trial], yield_all_future_ssp245_s[19:39,trial])
    matrix_percent_loss_num_year_ssp585_2050[:,trial] = get_num_year_with_loss(reference_yield_2001_2020_ssp585_state_flatten[trial], yield_all_future_ssp585_s[19:39,trial])

matrix_percent_loss_all_trial_ssp245_2090 = np.zeros((20,21,8000))
matrix_percent_loss_all_trial_ssp585_2090 = np.zeros((20,21,8000))
matrix_percent_loss_all_trial_ssp245_2050 = np.zeros((20,21,8000))
matrix_percent_loss_all_trial_ssp585_2050 = np.zeros((20,21,8000))
loss_percent_axis = np.linspace(0, 1, 21)
for trial in range(8000):
    for percent_loss in range(0,21):
        matrix_percent_loss_all_trial_ssp245_2090[:,percent_loss,trial] = (np.arange(1,21) <= matrix_percent_loss_num_year_ssp245_2090[percent_loss,trial])*1
        matrix_percent_loss_all_trial_ssp585_2090[:,percent_loss,trial] = (np.arange(1,21) <= matrix_percent_loss_num_year_ssp585_2090[percent_loss,trial])*1
        matrix_percent_loss_all_trial_ssp245_2050[:,percent_loss,trial] = (np.arange(1,21) <= matrix_percent_loss_num_year_ssp245_2050[percent_loss,trial])*1
        matrix_percent_loss_all_trial_ssp585_2050[:,percent_loss,trial] = (np.arange(1,21) <= matrix_percent_loss_num_year_ssp585_2050[percent_loss,trial])*1
        
prob_percent_loss_year_ssp245_2090 = np.mean(matrix_percent_loss_all_trial_ssp245_2090,axis=2)[::-1]
prob_percent_loss_year_ssp585_2090 = np.mean(matrix_percent_loss_all_trial_ssp585_2090,axis=2)[::-1]
prob_percent_loss_year_ssp245_2050 = np.mean(matrix_percent_loss_all_trial_ssp245_2050,axis=2)[::-1]
prob_percent_loss_year_ssp585_2050 = np.mean(matrix_percent_loss_all_trial_ssp585_2050,axis=2)[::-1]


fig = plt.figure()
fig.set_figheight(30)
fig.set_figwidth(40)
spec = gridspec.GridSpec(nrows=100, ncols=9, width_ratios=[3,0.05,0.15,0.15,0.15,1.5,1.5,0.4,1.5], wspace = 0,hspace = 0)

ax0 = plt.subplot(spec[5:35,0])
t = np.arange(2021,2100,1)
ax0=tsplot(t, np.transpose(yield_all_future_ssp245), color = 'royalblue')
plt.text(1966,2.6,'a',fontweight='bold', fontsize=35)
ax0.tick_params(axis = 'y', width=2, length = 5)
ax0.tick_params(axis = 'x', width=2, length = 5)
ax0_second_axis = ax0.secondary_yaxis("right")
ax0_second_axis.tick_params(axis="y", direction="out", length=5,width=2)
ax0_second_axis.set(yticklabels=[])
ax0.set(xlabel=None)
ax0.set(xticklabels=[])
plt.title('SSP245', fontsize = 35, y = 1.03)
plt.plot(t,np.median(yield_all_future_ssp245, axis=1), color = 'b', linestyle = 'solid', linewidth = 4)
tsplot(t, np.transpose(yield_all_future_ssp245_s), color = 'lightcoral')
plt.plot(t,np.median(yield_all_future_ssp245_s, axis=1), color = 'r',linestyle = 'solid',  linewidth = 4)
blue_patch = mpatches.Patch(color = 'royalblue', label = 'Sustained innovation')
red_patch = mpatches.Patch(color = 'lightcoral', label = 'Stopped innovation')
grey_patch = mpatches.Patch(color = 'grey', label = 'Historical simulations')
purple_patch = mpatches.Patch(color = 'mediumorchid', label = 'Decelerating innovation')
#blank_patch = mpatches.Patch(color = 'white', label = 'Future projections:')
plt.yticks(fontsize = 30)
plt.xlim(1980,2100)
plt.ylim(0,2.5)
plt.ylabel('Yield ton/acre', fontsize = 35)
plt.xticks((1980,1990,2000,2010,2020,2030,2040,2050,2060,2070,2080,2090,2100),['1980','1990', '2000', '2010','2020','2030', '2040', '2050','2060','2070', '2080','2090','2100'],fontsize = 30, rotation = 45)
plt.plot(np.arange(1980,2021), yield_observed_state[0:41], linewidth = 4, color = 'green')
tsplot(np.arange(1980,2021), np.transpose(yield_all_hist_ssp245[0:41]) , color = 'grey')
plt.plot(np.arange(1980,2021), np.median(yield_all_hist_ssp245, axis=1), color = 'black', linewidth = 4)
first_legend = plt.legend(handles=[Line2D([0], [0], color='green', lw=4, label='Observed yield'),grey_patch], fontsize = 30,fancybox=False, shadow=False, ncol = 1, bbox_to_anchor=(0.5, -1.8), edgecolor = 'white')
plt.gca().add_artist(first_legend)
second_legend = plt.legend(handles=[blue_patch, purple_patch, red_patch],fontsize = 30,fancybox=False, shadow=False, ncol = 1, bbox_to_anchor=(1.1, -1.8), edgecolor = 'white')
plt.gca().add_artist(second_legend)
#font = font_manager.FontProperties(weight='bold', size=30)
#plt.legend(handles=[blank_patch],fontsize = 30,fancybox=False, shadow=False, ncol = 1, bbox_to_anchor=(0.33, -1.35), edgecolor = 'white', prop = font)
box_legend = np.zeros((8000,5))
box_legend[:,0] = np.random.normal(2,0.16,size = (8000))
box_legend[:,1] = box_legend[:,0]
box_legend[:,2] = box_legend[:,0]
box_legend[:,3] = box_legend[:,0]
box_legend[:,4] = box_legend[:,0]
tsplot(np.array([1985,1986,1987,1988,1989]),box_legend, color = 'royalblue')
plt.plot(np.array([1985,1986,1987,1988,1989]), np.median(box_legend, axis = 0), linewidth = 4, linestyle = 'solid', color = 'b')
plt.text(x = 1990,y=1.8, s='67% CI', fontsize = 30)
plt.text(x = 1990,y=1.65, s='95% CI', fontsize = 30)
plt.text(x = 1990,y=1.95, s='Median', fontsize = 30)
#plt.plot(t,np.median(yield_all_future_ssp245_m, axis=1), color = 'purple', linestyle = 'solid', linewidth = 4)

ax4 = plt.subplot(spec[50:80,0])
ax4 = tsplot(t, np.transpose(yield_all_future_ssp585), color = 'royalblue')
plt.text(1966,2.6,'b',fontweight='bold', fontsize=35)
ax4.tick_params(axis = 'y', width=2, length = 5)
ax4.tick_params(axis = 'x', width=2, length = 5)
ax4_second_axis = ax4.secondary_yaxis("right")
ax4_second_axis.tick_params(axis="y", direction="out", length=5,width=2)
ax4_second_axis.set(yticklabels=[])
ax4.set(xlabel=None)
ax4.set(xticklabels=[])
plt.title('SSP585', fontsize = 35, y = 1.03)
plt.plot(t,np.median(yield_all_future_ssp585, axis=1), color = 'b', linestyle = 'solid', linewidth = 4)
tsplot(t, np.transpose(yield_all_future_ssp585_s), color = 'lightcoral')
plt.plot(t,np.median(yield_all_future_ssp585_s, axis=1), color = 'r',linestyle = 'solid', linewidth = 4)
plt.yticks(fontsize = 30)
plt.xlim(1980,2100)
plt.xlabel('Year', fontsize = 35)
plt.xticks((1980,1990,2000,2010,2020,2030,2040,2050,2060,2070,2080,2090,2100),['1980','1990', '2000', '2010','2020','2030', '2040', '2050','2060','2070', '2080','2090','2100'],fontsize = 30, rotation = 45)
plt.ylabel('Yield ton/acre', fontsize = 35)
plt.xlim(1980,2100)
plt.ylim(0,2.5)
plt.plot(np.arange(1980,2021), yield_observed_state[0:41], linewidth = 4, color = 'green')
tsplot(np.arange(1980,2021), y = np.transpose(yield_all_hist_ssp585[0:41]) , color = 'grey')
plt.plot(np.arange(1980,2021), np.median(yield_all_hist_ssp585, axis=1), color = 'black', linewidth = 4)
#plt.plot(t,np.median(yield_all_future_ssp585_m, axis=1), color = 'purple', linestyle = 'solid', linewidth = 4)
plt.subplots_adjust(bottom=0.2)


ax1_box = plt.subplot(spec[5:35,2])
#my_pal = {"yes": "royalblue",  "m": "mediumorchid", "no": "lightcoral"}
ax1_box = sns.boxplot(data = df_2080_2099_20yrmean_yield_ssp245, y = 'mean_yield', x = 'scenario',
              width = 0.8,fliersize = 0,  linewidth = 3, color = 'royalblue')
ax1_box.spines['top'].set_visible(False)
ax1_box.spines['right'].set_visible(False)
plt.yticks(fontsize =35)
ax1_box.set(xticklabels=[])
ax1_box.set(xlabel=None)
ax1_box.tick_params(left=False, bottom=False)
ax1_box.set(ylabel=None)
ax1_box.set(yticklabels=[])
plt.ylim(0,2.5)
ax1_box.spines['top'].set_visible(False)
ax1_box.spines['right'].set_visible(False)
ax1_box.spines['bottom'].set_visible(False)
ax1_box.spines['left'].set_visible(False)
ax11_box = plt.subplot(spec[5:35,3])
#my_pal = {"yes": "royalblue",  "m": "mediumorchid", "no": "lightcoral"}
ax11_box = sns.boxplot(data = df_2080_2099_20yrmean_yield_ssp245_m, y = 'mean_yield', x = 'scenario',
              width = 0.8,fliersize = 0,  linewidth = 3, color = 'mediumorchid')
ax11_box.spines['top'].set_visible(False)
ax11_box.spines['right'].set_visible(False)
plt.yticks(fontsize =35)
ax11_box.set(xticklabels=[])
ax11_box.set(xlabel=None)
ax11_box.tick_params(left=False, bottom=False)
ax11_box.set(ylabel=None)
ax11_box.set(yticklabels=[])
plt.ylim(0,2.5)
ax11_box.spines['top'].set_visible(False)
ax11_box.spines['right'].set_visible(False)
ax11_box.spines['bottom'].set_visible(False)
ax11_box.spines['left'].set_visible(False)

ax111_box = plt.subplot(spec[5:35,4])
#my_pal = {"yes": "royalblue",  "m": "mediumorchid", "no": "lightcoral"}
ax111_box = sns.boxplot(data = df_2080_2099_20yrmean_yield_ssp245_s, y = 'mean_yield', x = 'scenario',
              width = 0.8,fliersize = 0,  linewidth = 3, color = 'lightcoral')
ax111_box.spines['top'].set_visible(False)
ax111_box.spines['right'].set_visible(False)
plt.yticks(fontsize =35)
ax111_box.set(xticklabels=[])
ax111_box.set(xlabel=None)
ax111_box.tick_params(left=False, bottom=False)
ax111_box.set(ylabel=None)
ax111_box.set(yticklabels=[])
plt.ylim(0,2.5)
ax111_box.spines['top'].set_visible(False)
ax111_box.spines['right'].set_visible(False)
ax111_box.spines['bottom'].set_visible(False)
ax111_box.spines['left'].set_visible(False)

ax111_box.text(0.6, np.median(df_2080_2099_20yrmean_yield_ssp245_s.mean_yield)-0.07,s = str(np.around(np.median(df_2080_2099_20yrmean_yield_ssp245_s.mean_yield),1))+'['+
             str(np.round(yield_change_to_simulate2020_ssp245_s_ave,1))+'%]'
             , fontsize = 30, color = 'lightcoral')
ax111_box.text(0.6, np.median(df_2080_2099_20yrmean_yield_ssp245.mean_yield)-0.07,s = str(np.around(np.median(df_2080_2099_20yrmean_yield_ssp245.mean_yield),1))+'['+
             str(np.round(yield_change_to_simulate2020_ssp245_ave, 1))+'%]'
             , fontsize = 30, color = 'royalblue')
ax111_box.text(0.6, np.median(df_2080_2099_20yrmean_yield_ssp245_m.mean_yield)-0.07,s = str(np.around(np.median(df_2080_2099_20yrmean_yield_ssp245_m.mean_yield),1))+'['+
             str(np.round(yield_change_to_simulate2020_ssp245_m_ave, 1))+'%]'
             , fontsize = 30, color = 'mediumorchid')


ax6_box = plt.subplot(spec[50:80,2])
ax6_box = sns.boxplot(data = df_2080_2099_20yrmean_yield_ssp585, y = 'mean_yield', x = 'scenario',
              width = 0.8,fliersize = 0,  linewidth = 3,color = 'royalblue')
ax6_box.spines['top'].set_visible(False)
ax6_box.spines['right'].set_visible(False)
plt.yticks(fontsize =35)
ax6_box.set(xticklabels=[])
ax6_box.set(xlabel=None)
ax6_box.tick_params(left=False, bottom=False)
ax6_box.set(ylabel=None)
ax6_box.set(yticklabels=[])
plt.ylim(0,2.5)
ax6_box.spines['top'].set_visible(False)
ax6_box.spines['right'].set_visible(False)
ax6_box.spines['bottom'].set_visible(False)
ax6_box.spines['left'].set_visible(False)

ax66_box = plt.subplot(spec[50:80,3])
ax66_box = sns.boxplot(data = df_2080_2099_20yrmean_yield_ssp585_m, y = 'mean_yield', x = 'scenario',
              width = 0.8,fliersize = 0,  linewidth = 3, color = 'mediumorchid')
ax66_box.spines['top'].set_visible(False)
ax66_box.spines['right'].set_visible(False)
plt.yticks(fontsize =35)
ax66_box.set(xticklabels=[])
ax66_box.set(xlabel=None)
ax66_box.tick_params(left=False, bottom=False)
ax66_box.set(ylabel=None)
ax66_box.set(yticklabels=[])
plt.ylim(0,2.5)
ax66_box.spines['top'].set_visible(False)
ax66_box.spines['right'].set_visible(False)
ax66_box.spines['bottom'].set_visible(False)
ax66_box.spines['left'].set_visible(False)

ax666_box = plt.subplot(spec[50:80,4])
ax666_box = sns.boxplot(data = df_2080_2099_20yrmean_yield_ssp585_s, y = 'mean_yield', x = 'scenario',
              width = 0.8,fliersize = 0,  linewidth = 3, color = 'lightcoral')
ax666_box.spines['top'].set_visible(False)
ax666_box.spines['right'].set_visible(False)
plt.yticks(fontsize =35)
ax666_box.set(xticklabels=[])
ax666_box.set(xlabel=None)
ax666_box.tick_params(left=False, bottom=False)
ax666_box.set(ylabel=None)
ax666_box.set(yticklabels=[])
plt.ylim(0,2.5)
ax666_box.spines['top'].set_visible(False)
ax666_box.spines['right'].set_visible(False)
ax666_box.spines['bottom'].set_visible(False)
ax666_box.spines['left'].set_visible(False)

ax666_box.text(0.6, np.median(df_2080_2099_20yrmean_yield_ssp585_s.mean_yield)-0.07,s = str(np.around(np.median(df_2080_2099_20yrmean_yield_ssp585_s.mean_yield),1))+'['+
             str(np.round(yield_change_to_simulate2020_ssp585_s_ave,1))+'%]'
             , fontsize = 30, color = 'lightcoral')
ax666_box.text(0.6, np.median(df_2080_2099_20yrmean_yield_ssp585.mean_yield)-0.07,s = str(np.around(np.median(df_2080_2099_20yrmean_yield_ssp585.mean_yield),1))+'['+
             str(np.round(yield_change_to_simulate2020_ssp585_ave, 1))+'%]'
             , fontsize = 30, color = 'royalblue')
ax666_box.text(0.6, np.median(df_2080_2099_20yrmean_yield_ssp585_m.mean_yield)-0.07,s = str(np.around(np.median(df_2080_2099_20yrmean_yield_ssp585_m.mean_yield),1))+'['+
             str(np.round(yield_change_to_simulate2020_ssp585_m_ave, 1))+'%]'
             , fontsize = 30, color = 'mediumorchid')


ax8 = plt.subplot(spec[7:37,6])
ax8.imshow(prob_percent_loss_year_ssp245_2050, cmap = 'coolwarm')
plt.text(-4.1,-3,'c',fontweight='bold', fontsize=35)
ax8.tick_params(axis = 'y', width=2, length = 5)
ax8.tick_params(axis = 'x', width=2, length = 5)
ax8.contour(prob_percent_loss_year_ssp245_2050, levels=[0.9], colors='black', linewidths=5, linestyles = 'dashed')
ax8.contour(prob_percent_loss_year_ssp245_2050, levels=[0.1], colors='black', linewidths=5, linestyles = 'dotted')
ax8.contour(prob_percent_loss_year_ssp245_2050, levels=[0.5], colors='black', linewidths=5, linestyles = 'solid')
y_tick_pos = np.array([0,4,8,12,16,19])
y_tick_value = ('20','16','12','8','4','1')
plt.xticks(np.arange(0,21,4), np.array(np.linspace(0,100,6)).astype(int), rotation = 45)
plt.xticks(fontsize = 30)
plt.yticks(y_tick_pos, y_tick_value, fontsize = 30, rotation = 360)
plt.title('2040-2059', fontsize = 35, y = 1.05)
plt.ylabel('Number of years', fontsize = 35)
ax8.set(xlabel=None)
#ax8.set(xticklabels=[])

ax9 = plt.subplot(spec[7:37,8])
ax9.imshow(prob_percent_loss_year_ssp245_2090, cmap = 'coolwarm')
plt.text(21,-3,'d',fontweight='bold', fontsize=35)
ax9.tick_params(axis = 'y', width=2, length = 5)
ax9.tick_params(axis = 'x', width=2, length = 5)
ax9.contour(prob_percent_loss_year_ssp245_2090, levels=[0.9], colors='black', linewidths=5, linestyles = 'dashed')
ax9.contour(prob_percent_loss_year_ssp245_2090, levels=[0.1], colors='black', linewidths=5, linestyles = 'dotted')
ax9.contour(prob_percent_loss_year_ssp245_2090, levels=[0.5], colors='black', linewidths=5, linestyles = 'solid')
plt.xticks(np.arange(0,21,4), np.array(np.linspace(0,100,6)).astype(int), rotation = 45)
plt.xticks(fontsize = 30)
plt.yticks(y_tick_pos, y_tick_value, fontsize = 30, rotation = 360)
ax9.set(xlabel=None)
#ax9.set(xticklabels=[])
plt.title('2080-2099', fontsize = 35, y = 1.05)
plt.text(-6.8, -3.5, 'SSP245', fontsize = 35)


ax10 = plt.subplot(spec[51:100,6])
ax10.imshow(prob_percent_loss_year_ssp585_2050, cmap = 'coolwarm')
plt.text(-4.1,-3,'e',fontweight='bold', fontsize=35)
ax10.tick_params(axis = 'y', width=2, length = 5)
ax10.tick_params(axis = 'x', width=2, length = 5)
ax10.contour(prob_percent_loss_year_ssp585_2050, levels=[0.9], colors='black', linewidths=5, linestyles = 'dashed')
ax10.contour(prob_percent_loss_year_ssp585_2050, levels=[0.1], colors='black', linewidths=5, linestyles = 'dotted')
ax10.contour(prob_percent_loss_year_ssp585_2050, levels=[0.5], colors='black', linewidths=5, linestyles = 'solid')
plt.xticks(np.arange(0,21,4), np.array(np.linspace(0,100,6)).astype(int), rotation = 45)
plt.xticks(fontsize = 30)
plt.yticks(y_tick_pos, y_tick_value, fontsize = 30, rotation = 360)
plt.ylabel('Number of years', fontsize = 35)
ax10.set_xlabel('Percentage of yield loss from climate change', fontsize = 35)
ax10.xaxis.set_label_coords(1.1, -.2)
plt.title('2040-2059', fontsize = 35, y = 1.05)
#ax10.yaxis.set_label_coords(-0.1, 1.15)

ax11 = plt.subplot(spec[51:100,8])
im = ax11.imshow(prob_percent_loss_year_ssp585_2090, cmap = 'coolwarm')
plt.text(21,-3,'f',fontweight='bold', fontsize=35)
ax11.tick_params(axis = 'y', width=2, length = 5)
ax11.tick_params(axis = 'x', width=2, length = 5)
ax11.contour(prob_percent_loss_year_ssp585_2090, levels=[0.9], colors='black', linewidths=5, linestyles = 'dashed')
ax11.contour(prob_percent_loss_year_ssp585_2090, levels=[0.1], colors='black', linewidths=5, linestyles = 'dotted')
ax11.contour(prob_percent_loss_year_ssp585_2090, levels=[0.5], colors='black', linewidths=5, linestyles = 'solid')
plt.xticks(np.arange(0,21,4), np.array(np.linspace(0,100,6)).astype(int), rotation = 45)
plt.xticks(fontsize = 30)
plt.yticks(y_tick_pos, y_tick_value, fontsize = 30, rotation = 360)
cbar = plt.colorbar(im, ax = [ax10,ax11], location = 'bottom', pad = 0.26, shrink = 0.8, cmap = 'coolwarm')
cbar.ax.tick_params(labelsize=30)
cbar.set_label('Probability', fontsize = 35, labelpad=0.1)
plt.title('2080-2099', fontsize = 35, y = 1.05)
plt.text(-6.8, -3.5, 'SSP585', fontsize = 35)
plt.savefig(save_path+'yield_time_series.pdf', dpi = 300)


##Figure 3: yield time series for MACA
df_2080_2099_20yrmean_yield_rcp45 = pd.DataFrame({'scenario' : np.repeat('rcp45', 18000),'mean_yield' : np.mean(yield_all_future_rcp45[-20:],axis=0), 'tech' : np.repeat('yes',18000)})
df_2080_2099_20yrmean_yield_rcp45_s = pd.DataFrame({'scenario' : np.repeat('rcp45', 18000),'mean_yield' : np.mean(yield_all_future_rcp45_s[-20:],axis=0), 'tech' : np.repeat('no',18000)})
df_2080_2099_20yrmean_yield_rcp45_m = pd.DataFrame({'scenario' : np.repeat('rcp45', 18000),'mean_yield' : np.mean(yield_all_future_rcp45_m[-20:],axis=0), 'tech' : np.repeat('m',18000)})
df_2080_2099_20yrmean_yield_rcp85 = pd.DataFrame({'scenario' : np.repeat('rcp85', 18000),'mean_yield' : np.mean(yield_all_future_rcp85[-20:],axis=0), 'tech' : np.repeat('yes',18000)})
df_2080_2099_20yrmean_yield_rcp85_s = pd.DataFrame({'scenario' : np.repeat('rcp85', 18000),'mean_yield' : np.mean(yield_all_future_rcp85_s[-20:],axis=0), 'tech' : np.repeat('no',18000)})
df_2080_2099_20yrmean_yield_rcp85_m = pd.DataFrame({'scenario' : np.repeat('rcp85', 18000),'mean_yield' : np.mean(yield_all_future_rcp85_m[-20:],axis=0), 'tech' : np.repeat('m',18000)})
df_2080_2099_20yrmean_yield_rcp45_total = pd.concat((df_2080_2099_20yrmean_yield_rcp45, df_2080_2099_20yrmean_yield_rcp45_s, df_2080_2099_20yrmean_yield_rcp45_m))
df_2080_2099_20yrmean_yield_rcp85_total = pd.concat((df_2080_2099_20yrmean_yield_rcp85, df_2080_2099_20yrmean_yield_rcp85_s, df_2080_2099_20yrmean_yield_rcp85_m))

yield_change_to_simulate2020_rcp45 = ((np.mean(yield_all_future_rcp45[-20:],axis=0) - reference_yield_2001_2020_rcp45_state_flatten)*100/reference_yield_2001_2020_rcp45_state_flatten).reshape(18,1000)
yield_change_to_simulate2020_rcp85 = ((np.mean(yield_all_future_rcp85[-20:],axis=0) - reference_yield_2001_2020_rcp85_state_flatten)*100/reference_yield_2001_2020_rcp85_state_flatten).reshape(18,1000)
yield_change_to_simulate2020_rcp45_s = ((np.mean(yield_all_future_rcp45_s[-20:],axis=0) - reference_yield_2001_2020_rcp45_state_flatten)*100/reference_yield_2001_2020_rcp45_state_flatten).reshape(18,1000)
yield_change_to_simulate2020_rcp85_s = ((np.mean(yield_all_future_rcp85_s[-20:],axis=0) - reference_yield_2001_2020_rcp85_state_flatten)*100/reference_yield_2001_2020_rcp85_state_flatten).reshape(18,1000)
yield_change_to_simulate2020_rcp45_m = ((np.mean(yield_all_future_rcp45_m[-20:],axis=0) - reference_yield_2001_2020_rcp45_state_flatten)*100/reference_yield_2001_2020_rcp45_state_flatten).reshape(18,1000)
yield_change_to_simulate2020_rcp85_m = ((np.mean(yield_all_future_rcp85_m[-20:],axis=0) - reference_yield_2001_2020_rcp85_state_flatten)*100/reference_yield_2001_2020_rcp85_state_flatten).reshape(18,1000)

yield_change_to_simulate2020_rcp45_ave = np.median(np.mean(yield_change_to_simulate2020_rcp45, axis=0), axis = 0)
yield_change_to_simulate2020_rcp85_ave = np.median(np.mean(yield_change_to_simulate2020_rcp85, axis=0), axis = 0)
yield_change_to_simulate2020_rcp45_s_ave = np.median(np.mean(yield_change_to_simulate2020_rcp45_s, axis=0), axis = 0)
yield_change_to_simulate2020_rcp85_s_ave = np.median(np.mean(yield_change_to_simulate2020_rcp85_s, axis=0), axis = 0)
yield_change_to_simulate2020_rcp45_m_ave = np.median(np.mean(yield_change_to_simulate2020_rcp45_m, axis=0), axis = 0)
yield_change_to_simulate2020_rcp85_m_ave = np.median(np.mean(yield_change_to_simulate2020_rcp85_m, axis=0), axis = 0)

matrix_percent_loss_num_year_rcp45_2090 = np.zeros((21,18000))
matrix_percent_loss_num_year_rcp85_2090 = np.zeros((21,18000))
matrix_percent_loss_num_year_rcp45_2050 = np.zeros((21,18000))
matrix_percent_loss_num_year_rcp85_2050 = np.zeros((21,18000))
for trial in range(18000):
    matrix_percent_loss_num_year_rcp45_2090[:,trial] = get_num_year_with_loss(reference_yield_2001_2020_rcp45_state_flatten[trial], yield_all_future_rcp45_s[-20:,trial])
    matrix_percent_loss_num_year_rcp85_2090[:,trial] = get_num_year_with_loss(reference_yield_2001_2020_rcp85_state_flatten[trial], yield_all_future_rcp85_s[-20:,trial])
    matrix_percent_loss_num_year_rcp45_2050[:,trial] = get_num_year_with_loss(reference_yield_2001_2020_rcp45_state_flatten[trial], yield_all_future_rcp45_s[19:39,trial])
    matrix_percent_loss_num_year_rcp85_2050[:,trial] = get_num_year_with_loss(reference_yield_2001_2020_rcp85_state_flatten[trial], yield_all_future_rcp85_s[19:39,trial])

matrix_percent_loss_all_trial_rcp45_2090 = np.zeros((20,21,18000))
matrix_percent_loss_all_trial_rcp85_2090 = np.zeros((20,21,18000))
matrix_percent_loss_all_trial_rcp45_2050 = np.zeros((20,21,18000))
matrix_percent_loss_all_trial_rcp85_2050 = np.zeros((20,21,18000))
loss_percent_axis = np.linspace(0, 1, 21)
for trial in range(18000):
    for percent_loss in range(0,21):
        matrix_percent_loss_all_trial_rcp45_2090[:,percent_loss,trial] = (np.arange(1,21) <= matrix_percent_loss_num_year_rcp45_2090[percent_loss,trial])*1
        matrix_percent_loss_all_trial_rcp85_2090[:,percent_loss,trial] = (np.arange(1,21) <= matrix_percent_loss_num_year_rcp85_2090[percent_loss,trial])*1
        matrix_percent_loss_all_trial_rcp45_2050[:,percent_loss,trial] = (np.arange(1,21) <= matrix_percent_loss_num_year_rcp45_2050[percent_loss,trial])*1
        matrix_percent_loss_all_trial_rcp85_2050[:,percent_loss,trial] = (np.arange(1,21) <= matrix_percent_loss_num_year_rcp85_2050[percent_loss,trial])*1
        
prob_percent_loss_year_rcp45_2090 = np.mean(matrix_percent_loss_all_trial_rcp45_2090,axis=2)[::-1]
prob_percent_loss_year_rcp85_2090 = np.mean(matrix_percent_loss_all_trial_rcp85_2090,axis=2)[::-1]
prob_percent_loss_year_rcp45_2050 = np.mean(matrix_percent_loss_all_trial_rcp45_2050,axis=2)[::-1]
prob_percent_loss_year_rcp85_2050 = np.mean(matrix_percent_loss_all_trial_rcp85_2050,axis=2)[::-1]

fig = plt.figure()
fig.set_figheight(30)
fig.set_figwidth(40)
spec = gridspec.GridSpec(nrows=100, ncols=9, width_ratios=[3,0.05,0.15,0.15,0.15,1.5,1.5,0.4,1.5], wspace = 0,hspace = 0)

ax0 = plt.subplot(spec[5:35,0])
t = np.arange(2021,2100,1)
ax0=tsplot(t, np.transpose(yield_all_future_rcp45), color = 'royalblue')
plt.text(1966,2.6,'a',fontweight='bold', fontsize=35)
ax0.tick_params(axis = 'y', width=2, length = 5)
ax0.tick_params(axis = 'x', width=2, length = 5)
ax0_second_axis = ax0.secondary_yaxis("right")
ax0_second_axis.tick_params(axis="y", direction="out", length=5,width=2)
ax0_second_axis.set(yticklabels=[])
ax0.set(xlabel=None)
ax0.set(xticklabels=[])
plt.title('RCP4.5', fontsize = 35, y = 1.03)
plt.plot(t,np.median(yield_all_future_rcp45, axis=1), color = 'b', linestyle = 'solid', linewidth = 4)
tsplot(t, np.transpose(yield_all_future_rcp45_s), color = 'lightcoral')
plt.plot(t,np.median(yield_all_future_rcp45_s, axis=1), color = 'r',linestyle = 'solid',  linewidth = 4)
blue_patch = mpatches.Patch(color = 'royalblue', label = 'Sustained innovation')
red_patch = mpatches.Patch(color = 'lightcoral', label = 'Stopped innovation')
grey_patch = mpatches.Patch(color = 'grey', label = 'Historical simulations')
purple_patch = mpatches.Patch(color = 'mediumorchid', label = 'Decelerating innovation')
#blank_patch = mpatches.Patch(color = 'white', label = 'Future projections:')
plt.yticks(fontsize = 30)
plt.xlim(1980,2100)
plt.ylim(0,2.5)
plt.ylabel('Yield ton/acre', fontsize = 35)
plt.xticks((1980,1990,2000,2010,2020,2030,2040,2050,2060,2070,2080,2090,2100),['1980','1990', '2000', '2010','2020','2030', '2040', '2050','2060','2070', '2080','2090','2100'],fontsize = 30, rotation = 45)
plt.plot(np.arange(1980,2021), yield_observed_state[0:41], linewidth = 4, color = 'green')
tsplot(np.arange(1980,2021), np.transpose(yield_all_hist_rcp45[0:41]) , color = 'grey')
plt.plot(np.arange(1980,2021), np.median(yield_all_hist_rcp45, axis=1), color = 'black', linewidth = 4)
first_legend = plt.legend(handles=[Line2D([0], [0], color='green', lw=4, label='Observed yield'),grey_patch], fontsize = 30,fancybox=False, shadow=False, ncol = 1, bbox_to_anchor=(0.5, -1.8), edgecolor = 'white')
plt.gca().add_artist(first_legend)
second_legend = plt.legend(handles=[blue_patch, purple_patch, red_patch],fontsize = 30,fancybox=False, shadow=False, ncol = 1, bbox_to_anchor=(1.1, -1.8), edgecolor = 'white')
plt.gca().add_artist(second_legend)
#font = font_manager.FontProperties(weight='bold', size=30)
#plt.legend(handles=[blank_patch],fontsize = 30,fancybox=False, shadow=False, ncol = 1, bbox_to_anchor=(0.33, -1.35), edgecolor = 'white', prop = font)
box_legend = np.zeros((18000,5))
box_legend[:,0] = np.random.normal(2,0.16,size = (18000))
box_legend[:,1] = box_legend[:,0]
box_legend[:,2] = box_legend[:,0]
box_legend[:,3] = box_legend[:,0]
box_legend[:,4] = box_legend[:,0]
tsplot(np.array([1985,1986,1987,1988,1989]),box_legend, color = 'royalblue')
plt.plot(np.array([1985,1986,1987,1988,1989]), np.median(box_legend, axis = 0), linewidth = 4, linestyle = 'solid', color = 'b')
plt.text(x = 1990,y=1.8, s='67% CI', fontsize = 30)
plt.text(x = 1990,y=1.65, s='95% CI', fontsize = 30)
plt.text(x = 1990,y=1.95, s='Median', fontsize = 30)
#plt.plot(t,np.median(yield_all_future_rcp45_m, axis=1), color = 'purple', linestyle = 'solid', linewidth = 4)

ax4 = plt.subplot(spec[50:80,0])
ax4 = tsplot(t, np.transpose(yield_all_future_rcp85), color = 'royalblue')
plt.text(1966,2.6,'b',fontweight='bold', fontsize=35)
ax4.tick_params(axis = 'y', width=2, length = 5)
ax4.tick_params(axis = 'x', width=2, length = 5)
ax4_second_axis = ax4.secondary_yaxis("right")
ax4_second_axis.tick_params(axis="y", direction="out", length=5,width=2)
ax4_second_axis.set(yticklabels=[])
ax4.set(xlabel=None)
ax4.set(xticklabels=[])
plt.title('RCP8.5', fontsize = 35, y = 1.03)
plt.plot(t,np.median(yield_all_future_rcp85, axis=1), color = 'b', linestyle = 'solid', linewidth = 4)
tsplot(t, np.transpose(yield_all_future_rcp85_s), color = 'lightcoral')
plt.plot(t,np.median(yield_all_future_rcp85_s, axis=1), color = 'r',linestyle = 'solid', linewidth = 4)
plt.yticks(fontsize = 30)
plt.xlim(1980,2100)
plt.xlabel('Year', fontsize = 35)
plt.xticks((1980,1990,2000,2010,2020,2030,2040,2050,2060,2070,2080,2090,2100),['1980','1990', '2000', '2010','2020','2030', '2040', '2050','2060','2070', '2080','2090','2100'],fontsize = 30, rotation = 45)
plt.ylabel('Yield ton/acre', fontsize = 35)
plt.xlim(1980,2100)
plt.ylim(0,2.5)
plt.plot(np.arange(1980,2021), yield_observed_state[0:41], linewidth = 4, color = 'green')
tsplot(np.arange(1980,2021), y = np.transpose(yield_all_hist_rcp85[0:41]) , color = 'grey')
plt.plot(np.arange(1980,2021), np.median(yield_all_hist_rcp85, axis=1), color = 'black', linewidth = 4)
#plt.plot(t,np.median(yield_all_future_rcp85_m, axis=1), color = 'purple', linestyle = 'solid', linewidth = 4)
plt.subplots_adjust(bottom=0.2)


ax1_box = plt.subplot(spec[5:35,2])
#my_pal = {"yes": "royalblue",  "m": "mediumorchid", "no": "lightcoral"}
ax1_box = sns.boxplot(data = df_2080_2099_20yrmean_yield_rcp45, y = 'mean_yield', x = 'scenario',
              width = 0.8,fliersize = 0,  linewidth = 3, color = 'royalblue')
ax1_box.spines['top'].set_visible(False)
ax1_box.spines['right'].set_visible(False)
plt.yticks(fontsize =35)
ax1_box.set(xticklabels=[])
ax1_box.set(xlabel=None)
ax1_box.tick_params(left=False, bottom=False)
ax1_box.set(ylabel=None)
ax1_box.set(yticklabels=[])
plt.ylim(0,2.5)
ax1_box.spines['top'].set_visible(False)
ax1_box.spines['right'].set_visible(False)
ax1_box.spines['bottom'].set_visible(False)
ax1_box.spines['left'].set_visible(False)
ax11_box = plt.subplot(spec[5:35,3])
#my_pal = {"yes": "royalblue",  "m": "mediumorchid", "no": "lightcoral"}
ax11_box = sns.boxplot(data = df_2080_2099_20yrmean_yield_rcp45_m, y = 'mean_yield', x = 'scenario',
              width = 0.8,fliersize = 0,  linewidth = 3, color = 'mediumorchid')
ax11_box.spines['top'].set_visible(False)
ax11_box.spines['right'].set_visible(False)
plt.yticks(fontsize =35)
ax11_box.set(xticklabels=[])
ax11_box.set(xlabel=None)
ax11_box.tick_params(left=False, bottom=False)
ax11_box.set(ylabel=None)
ax11_box.set(yticklabels=[])
plt.ylim(0,2.5)
ax11_box.spines['top'].set_visible(False)
ax11_box.spines['right'].set_visible(False)
ax11_box.spines['bottom'].set_visible(False)
ax11_box.spines['left'].set_visible(False)

ax111_box = plt.subplot(spec[5:35,4])
#my_pal = {"yes": "royalblue",  "m": "mediumorchid", "no": "lightcoral"}
ax111_box = sns.boxplot(data = df_2080_2099_20yrmean_yield_rcp45_s, y = 'mean_yield', x = 'scenario',
              width = 0.8,fliersize = 0,  linewidth = 3, color = 'lightcoral')
ax111_box.spines['top'].set_visible(False)
ax111_box.spines['right'].set_visible(False)
plt.yticks(fontsize =35)
ax111_box.set(xticklabels=[])
ax111_box.set(xlabel=None)
ax111_box.tick_params(left=False, bottom=False)
ax111_box.set(ylabel=None)
ax111_box.set(yticklabels=[])
plt.ylim(0,2.5)
ax111_box.spines['top'].set_visible(False)
ax111_box.spines['right'].set_visible(False)
ax111_box.spines['bottom'].set_visible(False)
ax111_box.spines['left'].set_visible(False)

ax111_box.text(0.6, np.median(df_2080_2099_20yrmean_yield_rcp45_s.mean_yield)-0.07,s = str(np.around(np.median(df_2080_2099_20yrmean_yield_rcp45_s.mean_yield),1))+'['+
             str(np.round(yield_change_to_simulate2020_rcp45_s_ave,1))+'%]'
             , fontsize = 30, color = 'lightcoral')
ax111_box.text(0.6, np.median(df_2080_2099_20yrmean_yield_rcp45.mean_yield)-0.07,s = str(np.around(np.median(df_2080_2099_20yrmean_yield_rcp45.mean_yield),1))+'['+
             str(np.round(yield_change_to_simulate2020_rcp45_ave, 1))+'%]'
             , fontsize = 30, color = 'royalblue')
ax111_box.text(0.6, np.median(df_2080_2099_20yrmean_yield_rcp45_m.mean_yield)-0.07,s = str(np.around(np.median(df_2080_2099_20yrmean_yield_rcp45_m.mean_yield),1))+'['+
             str(np.round(yield_change_to_simulate2020_rcp45_m_ave, 1))+'%]'
             , fontsize = 30, color = 'mediumorchid')


ax6_box = plt.subplot(spec[50:80,2])
ax6_box = sns.boxplot(data = df_2080_2099_20yrmean_yield_rcp85, y = 'mean_yield', x = 'scenario',
              width = 0.8,fliersize = 0,  linewidth = 3,color = 'royalblue')
ax6_box.spines['top'].set_visible(False)
ax6_box.spines['right'].set_visible(False)
plt.yticks(fontsize =35)
ax6_box.set(xticklabels=[])
ax6_box.set(xlabel=None)
ax6_box.tick_params(left=False, bottom=False)
ax6_box.set(ylabel=None)
ax6_box.set(yticklabels=[])
plt.ylim(0,2.5)
ax6_box.spines['top'].set_visible(False)
ax6_box.spines['right'].set_visible(False)
ax6_box.spines['bottom'].set_visible(False)
ax6_box.spines['left'].set_visible(False)

ax66_box = plt.subplot(spec[50:80,3])
ax66_box = sns.boxplot(data = df_2080_2099_20yrmean_yield_rcp85_m, y = 'mean_yield', x = 'scenario',
              width = 0.8,fliersize = 0,  linewidth = 3, color = 'mediumorchid')
ax66_box.spines['top'].set_visible(False)
ax66_box.spines['right'].set_visible(False)
plt.yticks(fontsize =35)
ax66_box.set(xticklabels=[])
ax66_box.set(xlabel=None)
ax66_box.tick_params(left=False, bottom=False)
ax66_box.set(ylabel=None)
ax66_box.set(yticklabels=[])
plt.ylim(0,2.5)
ax66_box.spines['top'].set_visible(False)
ax66_box.spines['right'].set_visible(False)
ax66_box.spines['bottom'].set_visible(False)
ax66_box.spines['left'].set_visible(False)

ax666_box = plt.subplot(spec[50:80,4])
ax666_box = sns.boxplot(data = df_2080_2099_20yrmean_yield_rcp85_s, y = 'mean_yield', x = 'scenario',
              width = 0.8,fliersize = 0,  linewidth = 3, color = 'lightcoral')
ax666_box.spines['top'].set_visible(False)
ax666_box.spines['right'].set_visible(False)
plt.yticks(fontsize =35)
ax666_box.set(xticklabels=[])
ax666_box.set(xlabel=None)
ax666_box.tick_params(left=False, bottom=False)
ax666_box.set(ylabel=None)
ax666_box.set(yticklabels=[])
plt.ylim(0,2.5)
ax666_box.spines['top'].set_visible(False)
ax666_box.spines['right'].set_visible(False)
ax666_box.spines['bottom'].set_visible(False)
ax666_box.spines['left'].set_visible(False)

ax666_box.text(0.6, np.median(df_2080_2099_20yrmean_yield_rcp85_s.mean_yield)-0.07,s = str(np.around(np.median(df_2080_2099_20yrmean_yield_rcp85_s.mean_yield),1))+'['+
             str(np.round(yield_change_to_simulate2020_rcp85_s_ave,1))+'%]'
             , fontsize = 30, color = 'lightcoral')
ax666_box.text(0.6, np.median(df_2080_2099_20yrmean_yield_rcp85.mean_yield)-0.07,s = str(np.around(np.median(df_2080_2099_20yrmean_yield_rcp85.mean_yield),1))+'['+
             str(np.round(yield_change_to_simulate2020_rcp85_ave, 1))+'%]'
             , fontsize = 30, color = 'royalblue')
ax666_box.text(0.6, np.median(df_2080_2099_20yrmean_yield_rcp85_m.mean_yield)-0.07,s = str(np.around(np.median(df_2080_2099_20yrmean_yield_rcp85_m.mean_yield),1))+'['+
             str(np.round(yield_change_to_simulate2020_rcp85_m_ave, 1))+'%]'
             , fontsize = 30, color = 'mediumorchid')


ax8 = plt.subplot(spec[7:37,6])
ax8.imshow(prob_percent_loss_year_rcp45_2050, cmap = 'coolwarm')
plt.text(-4.1,-3,'c',fontweight='bold', fontsize=35)
ax8.tick_params(axis = 'y', width=2, length = 5)
ax8.tick_params(axis = 'x', width=2, length = 5)
ax8.contour(prob_percent_loss_year_rcp45_2050, levels=[0.9], colors='black', linewidths=5, linestyles = 'dashed')
ax8.contour(prob_percent_loss_year_rcp45_2050, levels=[0.1], colors='black', linewidths=5, linestyles = 'dotted')
ax8.contour(prob_percent_loss_year_rcp45_2050, levels=[0.5], colors='black', linewidths=5, linestyles = 'solid')
y_tick_pos = np.array([0,4,8,12,16,19])
y_tick_value = ('20','16','12','8','4','1')
plt.xticks(np.arange(0,21,4), np.array(np.linspace(0,100,6)).astype(int), rotation = 45)
plt.xticks(fontsize = 30)
plt.yticks(y_tick_pos, y_tick_value, fontsize = 30, rotation = 360)
plt.title('2040-2059', fontsize = 35, y = 1.05)
plt.ylabel('Number of years', fontsize = 35)
ax8.set(xlabel=None)
#ax8.set(xticklabels=[])

ax9 = plt.subplot(spec[7:37,8])
ax9.imshow(prob_percent_loss_year_rcp45_2090, cmap = 'coolwarm')
plt.text(21,-3,'d',fontweight='bold', fontsize=35)
ax9.tick_params(axis = 'y', width=2, length = 5)
ax9.tick_params(axis = 'x', width=2, length = 5)
ax9.contour(prob_percent_loss_year_rcp45_2090, levels=[0.9], colors='black', linewidths=5, linestyles = 'dashed')
ax9.contour(prob_percent_loss_year_rcp45_2090, levels=[0.1], colors='black', linewidths=5, linestyles = 'dotted')
ax9.contour(prob_percent_loss_year_rcp45_2090, levels=[0.5], colors='black', linewidths=5, linestyles = 'solid')
plt.xticks(np.arange(0,21,4), np.array(np.linspace(0,100,6)).astype(int), rotation = 45)
plt.xticks(fontsize = 30)
plt.yticks(y_tick_pos, y_tick_value, fontsize = 30, rotation = 360)
ax9.set(xlabel=None)
#ax9.set(xticklabels=[])
plt.title('2080-2099', fontsize = 35, y = 1.05)
plt.text(-6.8, -3.5, 'RCP4.5', fontsize = 35)


ax10 = plt.subplot(spec[51:100,6])
ax10.imshow(prob_percent_loss_year_rcp85_2050, cmap = 'coolwarm')
plt.text(-4.1,-3,'e',fontweight='bold', fontsize=35)
ax10.tick_params(axis = 'y', width=2, length = 5)
ax10.tick_params(axis = 'x', width=2, length = 5)
ax10.contour(prob_percent_loss_year_rcp85_2050, levels=[0.9], colors='black', linewidths=5, linestyles = 'dashed')
ax10.contour(prob_percent_loss_year_rcp85_2050, levels=[0.1], colors='black', linewidths=5, linestyles = 'dotted')
ax10.contour(prob_percent_loss_year_rcp85_2050, levels=[0.5], colors='black', linewidths=5, linestyles = 'solid')
plt.xticks(np.arange(0,21,4), np.array(np.linspace(0,100,6)).astype(int), rotation = 45)
plt.xticks(fontsize = 30)
plt.yticks(y_tick_pos, y_tick_value, fontsize = 30, rotation = 360)
plt.ylabel('Number of years', fontsize = 35)
ax10.set_xlabel('Percentage of yield loss from climate change', fontsize = 35)
ax10.xaxis.set_label_coords(1.1, -.2)
plt.title('2040-2059', fontsize = 35, y = 1.05)
#ax10.yaxis.set_label_coords(-0.1, 1.15)

ax11 = plt.subplot(spec[51:100,8])
im = ax11.imshow(prob_percent_loss_year_rcp85_2090, cmap = 'coolwarm')
plt.text(21,-3,'f',fontweight='bold', fontsize=35)
ax11.tick_params(axis = 'y', width=2, length = 5)
ax11.tick_params(axis = 'x', width=2, length = 5)
ax11.contour(prob_percent_loss_year_rcp85_2090, levels=[0.9], colors='black', linewidths=5, linestyles = 'dashed')
ax11.contour(prob_percent_loss_year_rcp85_2090, levels=[0.1], colors='black', linewidths=5, linestyles = 'dotted')
ax11.contour(prob_percent_loss_year_rcp85_2090, levels=[0.5], colors='black', linewidths=5, linestyles = 'solid')
plt.xticks(np.arange(0,21,4), np.array(np.linspace(0,100,6)).astype(int), rotation = 45)
plt.xticks(fontsize = 30)
plt.yticks(y_tick_pos, y_tick_value, fontsize = 30, rotation = 360)
cbar = plt.colorbar(im, ax = [ax10,ax11], location = 'bottom', pad = 0.26, shrink = 0.8, cmap = 'coolwarm')
cbar.ax.tick_params(labelsize=30)
cbar.set_label('Probability', fontsize = 35, labelpad=0.1)
plt.title('2080-2099', fontsize = 35, y = 1.05)
plt.text(-6.8, -3.5, 'RCP8.5', fontsize = 35)
plt.savefig(save_path+'yield_time_series_MACA.pdf', dpi = 300)


## Figure 4: MAP
climate_change_only_average_county_yield_1980_2000_rcp85 = np.mean(np.sum(aci_contribution_rcp85_county_total, axis = 4)[:,:,0:20,:],axis = 2)
climate_change_only_average_county_yield_2000_2020_rcp85 = np.mean(np.sum(aci_contribution_rcp85_county_total, axis = 4)[:,:,20:40,:],axis = 2)
for i in range(16):
    county_constant = coef_sum[:,aci_num*2+16+i]
    for j in range(18):
        climate_change_only_average_county_yield_1980_2000_rcp85[i,j,:] = climate_change_only_average_county_yield_1980_2000_rcp85[i,j,:] + county_constant
        climate_change_only_average_county_yield_2000_2020_rcp85[i,j,:] = climate_change_only_average_county_yield_2000_2020_rcp85[i,j,:] + county_constant
climate_change_only_average_county_yield_hist_change_percent_rcp85 = 100 * np.median(np.mean((climate_change_only_average_county_yield_2000_2020_rcp85 - climate_change_only_average_county_yield_1980_2000_rcp85)/climate_change_only_average_county_yield_1980_2000_rcp85, axis=1), axis=1)
climate_change_only_average_county_yield_1980_2000_rcp85 = climate_change_only_average_county_yield_1980_2000_rcp85.reshape(16,18000)
climate_change_only_average_county_yield_2000_2020_rcp85 = climate_change_only_average_county_yield_2000_2020_rcp85.reshape(16,18000)
climate_change_only_average_county_yield_hist_change_value_rcp85 = (climate_change_only_average_county_yield_2000_2020_rcp85 - climate_change_only_average_county_yield_1980_2000_rcp85)

for i in range(0,16):
    locals()[str(county_list[i])+'yield_rcp85_s'] = np.row_stack((np.split(yield_all_model_hist_rcp85_s, 16)[i], np.split(yield_all_model_future_rcp85_s, 16)[i]))
    locals()[str(county_list[i])+'yield_rcp45_s'] = np.row_stack((np.split(yield_all_model_hist_rcp45_s, 16)[i], np.split(yield_all_model_future_rcp45_s, 16)[i]))
for i in range(0,16):
    locals()[str(county_list[i])+'county_yield_change_2020_rcp85'] = np.zeros((18000))
    locals()[str(county_list[i])+'county_yield_change_2020_rcp85'][:] = np.mean(reference_yield_2001_2020_rcp85[:,:,i,:],axis=2).reshape(18000)
    locals()[str(county_list[i])+'county_yield_change_2099_rcp85'] = np.zeros((18000,2))
    locals()[str(county_list[i])+'county_yield_change_2099_rcp85'][:,0] = 100 * ((np.nanmean(locals()[str(county_list[i])+'yield_rcp85_s'][100:120,:], axis=0))-locals()[str(county_list[i])+'county_yield_change_2020_rcp85'][:])/locals()[str(county_list[i])+'county_yield_change_2020_rcp85'][:]
    locals()[str(county_list[i])+'county_yield_change_2099_rcp85'][:,1] = ((np.nanmean(locals()[str(county_list[i])+'yield_rcp85_s'][100:120,:], axis=0))-locals()[str(county_list[i])+'county_yield_change_2020_rcp85'][:])
    locals()[str(county_list[i])+'county_yield_change_2020_rcp45'] = np.zeros((18000))
    locals()[str(county_list[i])+'county_yield_change_2020_rcp45'][:] = np.mean(reference_yield_2001_2020_rcp45[:,:,i,:],axis=2).reshape(18000)
    locals()[str(county_list[i])+'county_yield_change_2099_rcp45'] = np.zeros((18000,2))
    locals()[str(county_list[i])+'county_yield_change_2099_rcp45'][:,0] = 100 * ((np.nanmean(locals()[str(county_list[i])+'yield_rcp45_s'][100:120,:], axis=0))-locals()[str(county_list[i])+'county_yield_change_2020_rcp45'][:])/locals()[str(county_list[i])+'county_yield_change_2020_rcp45'][:]
    locals()[str(county_list[i])+'county_yield_change_2099_rcp45'][:,1] = ((np.nanmean(locals()[str(county_list[i])+'yield_rcp45_s'][100:120,:], axis=0))-locals()[str(county_list[i])+'county_yield_change_2020_rcp45'][:])
    
    locals()[str(county_list[i])+'county_con_tech_change_2099'] = np.zeros((1000,2))
    locals()[str(county_list[i])+'county_con_tech_change_2099'][:,0] = 100 * (np.mean(future_tech_trend_county_rcp45_con[100:120,i,:], axis=0) - np.mean(locals()[str(county_list[i])+'county_yield_change_2020_rcp45'][:].reshape(18,1000), axis=0)) / np.mean(locals()[str(county_list[i])+'county_yield_change_2020_rcp45'][:].reshape(18,1000), axis=0)
    locals()[str(county_list[i])+'county_con_tech_change_2099'][:,1] = (np.mean(future_tech_trend_county_rcp45_con[100:120,i,:], axis=0) - np.mean(locals()[str(county_list[i])+'county_yield_change_2020_rcp45'][:].reshape(18,1000), axis=0))
    locals()[str(county_list[i])+'county_int_tech_change_2099'] = np.zeros((1000,2))
    locals()[str(county_list[i])+'county_int_tech_change_2099'][:,0] = 100 * (np.mean(future_tech_trend_county_rcp45_int[100:120,i,:], axis=0) - np.mean(locals()[str(county_list[i])+'county_yield_change_2020_rcp45'][:].reshape(18,1000), axis=0)) / np.mean(locals()[str(county_list[i])+'county_yield_change_2020_rcp45'][:].reshape(18,1000), axis=0)
    locals()[str(county_list[i])+'county_int_tech_change_2099'][:,1] = (np.mean(future_tech_trend_county_rcp45_int[100:120,i,:], axis=0) - np.mean(locals()[str(county_list[i])+'county_yield_change_2020_rcp45'][:].reshape(18,1000), axis=0))

    locals()[str(county_list[i])+'county_tech_change_hist'] = np.zeros((1000,2))
    locals()[str(county_list[i])+'county_tech_change_hist'][:,0] = 100 * (np.mean(future_tech_trend_county_rcp45_con[20:40,i,:], axis=0) - np.mean(future_tech_trend_county_rcp45_con[0:20,i,:], axis=0)) / np.mean(future_tech_trend_county_rcp45_con[20:40,i,:], axis=0)
    locals()[str(county_list[i])+'county_tech_change_hist'][:,1] = (np.mean(future_tech_trend_county_rcp45_con[20:40,i,:], axis=0) - np.mean(future_tech_trend_county_rcp45_con[0:20,i,:], axis=0)) 


 
median_yield_change_2099_percent_rcp85 = np.zeros((16))
median_yield_change_2099_percent_rcp45 = np.zeros((16))
median_yield_change_hist_percent_rcp85 = np.zeros((16))

for i in range(0,16):
    median_yield_change_2099_percent_rcp85[i] = np.median(np.mean((locals()[str(county_list[i])+'county_yield_change_2099_rcp85'][:,0].reshape(18,1000)), axis = 0))
    median_yield_change_2099_percent_rcp45[i] = np.median(np.mean((locals()[str(county_list[i])+'county_yield_change_2099_rcp45'][:,0].reshape(18,1000)), axis = 0))
    median_yield_change_hist_percent_rcp85[i] = climate_change_only_average_county_yield_hist_change_percent_rcp85[i]

yield_change_for_shp_85_2099_percent = np.zeros((58))
yield_change_for_shp_85_2099_percent[:] = np.nan
yield_change_for_shp_45_2099_percent = np.zeros((58))
yield_change_for_shp_45_2099_percent[:] = np.nan
yield_change_for_shp_hist_percent_rcp85 = np.zeros((58))
yield_change_for_shp_hist_percent_rcp85[:] = np.nan


county_order_N_S = ['Tehama', 'Butte', 'Glenn', 'Yuba', 'Sutter', 'Colusa', 'Yolo', 'Solano', 'San Joaquin', 'Stanislaus', 'Madera', 'Merced', 'Fresno', 'Tulare', 'Kings', 'Kern']

N_S_order = np.zeros((16))
ca_MACA = geopandas.read_file(shp_path+'CA_Counties_TIGER2016.shp')
ca_county_remove_shp_MACA = geopandas.read_file(shp_path+'CA_Counties_TIGER2016.shp')
ca_county_remove = ['Sierra', 'Sacramento', 'Santa Barbara', 'Calaveras', 'Ventura','Los Angeles', 'Sonoma', 'San Diego', 'Placer', 'San Francisco', 'Marin', 'Mariposa', 'Lassen', 'Napa',
                    'Shasta', 'Monterey','Trinity', 'Mendocino', 'Inyo', 'Mono', 'Tuolumne', 'San Bernardino', 'Contra Costa', 'Alpine', 'El Dorado', 'San Benito', 'Humboldt','Riverside',
                    'Del Norte', 'Modoc', 'Santa Clara', 'Alameda', 'Nevada', 'Orange', 'Imperial', 'Amador', 'Lake', 'Plumas', 'San Mateo', 'Siskiyou', 'Santa Cruz','San Luis Obispo']
for i in range(0,len(ca_county_remove)):
    ca_county_remove_shp_MACA.drop(ca_county_remove_shp_MACA.index[ca_county_remove_shp_MACA['NAME']==ca_county_remove[i]], inplace=True)

for i in range(0,58):
    for index in range(0,16):
        if county_list[index] == ca_MACA.NAME[i]:
            yield_change_for_shp_85_2099_percent[i] = median_yield_change_2099_percent_rcp85[index]
            yield_change_for_shp_45_2099_percent[i] = median_yield_change_2099_percent_rcp45[index]
            yield_change_for_shp_hist_percent_rcp85[i] = median_yield_change_hist_percent_rcp85[index]

for i in range(0,16):
    N_S_order[np.array(np.where(ca_county_remove_shp_MACA['NAME'] == county_order_N_S[i]))] = i+1
ca_county_remove_shp_MACA['N_S_order'] = N_S_order.astype(int)

yield_change_for_shp_85_2099_df = pd.DataFrame({'NAME' : ca_MACA.NAME, 'rcp85_2099_percent' : yield_change_for_shp_85_2099_percent})
ca_merge_rcp85_2099 =  ca_MACA.merge(yield_change_for_shp_85_2099_df, on = 'NAME')
yield_change_for_shp_45_2099_df = pd.DataFrame({'NAME' : ca_MACA.NAME, 'rcp45_2099_percent' : yield_change_for_shp_45_2099_percent})
ca_merge_rcp45_2099 =  ca_MACA.merge(yield_change_for_shp_45_2099_df, on = 'NAME')
yield_change_for_shp_hist_percent_df_rcp85 = pd.DataFrame({'NAME' : ca_MACA.NAME, 'hist_change_percent' : yield_change_for_shp_hist_percent_rcp85})
ca_merge_hist_change_rcp85 = ca_MACA.merge(yield_change_for_shp_hist_percent_df_rcp85, on = 'NAME')


df_county_yield_rcp85_change = pd.DataFrame()
df_county_yield_rcp45_change = pd.DataFrame()
df_county_yield_hist_change_rcp85 = pd.DataFrame()



for i in range(0,16):
    df_county_yield_rcp85_change_ind = pd.DataFrame({'County' : str(county_list[i]) , 'Yield Change by 2099' : locals()[str(county_list[i])+'county_yield_change_2099_rcp85'][:,1]})
    df_county_yield_rcp85_change = pd.concat((df_county_yield_rcp85_change, df_county_yield_rcp85_change_ind))
    df_county_yield_rcp45_change_ind = pd.DataFrame({'County' : str(county_list[i]) , 'Yield Change by 2099' : locals()[str(county_list[i])+'county_yield_change_2099_rcp45'][:,1]})
    df_county_yield_rcp45_change = pd.concat((df_county_yield_rcp45_change, df_county_yield_rcp45_change_ind))
    df_county_yield_hist_change_ind_rcp85 = pd.DataFrame({'County' : str(county_list[i]) , 'Hist yield change' : climate_change_only_average_county_yield_hist_change_value_rcp85[i]})
    df_county_yield_hist_change_rcp85 = pd.concat((df_county_yield_hist_change_rcp85, df_county_yield_hist_change_ind_rcp85))
    
climate_change_only_average_county_yield_1980_2000 = np.mean(np.sum(aci_contribution_ssp585_county_total, axis = 4)[:,:,0:20,:],axis = 2)
climate_change_only_average_county_yield_2000_2020 = np.mean(np.sum(aci_contribution_ssp585_county_total, axis = 4)[:,:,20:40,:],axis = 2)
for i in range(16):
    county_constant = coef_sum[:,aci_num*2+16+i]
    for j in range(8):
        climate_change_only_average_county_yield_1980_2000[i,j,:] = climate_change_only_average_county_yield_1980_2000[i,j,:] + county_constant
        climate_change_only_average_county_yield_2000_2020[i,j,:] = climate_change_only_average_county_yield_2000_2020[i,j,:] + county_constant
climate_change_only_average_county_yield_hist_change_percent = 100 * np.median(np.mean((climate_change_only_average_county_yield_2000_2020 - climate_change_only_average_county_yield_1980_2000)/climate_change_only_average_county_yield_1980_2000, axis=1), axis=1)
climate_change_only_average_county_yield_1980_2000 = climate_change_only_average_county_yield_1980_2000.reshape(16,8000)
climate_change_only_average_county_yield_2000_2020 = climate_change_only_average_county_yield_2000_2020.reshape(16,8000)
climate_change_only_average_county_yield_hist_change_value = (climate_change_only_average_county_yield_2000_2020 - climate_change_only_average_county_yield_1980_2000)

for i in range(0,16):
    locals()[str(county_list[i])+'yield_ssp585_s'] = np.row_stack((np.split(yield_all_model_hist_ssp585_s, 16)[i], np.split(yield_all_model_future_ssp585_s, 16)[i]))
    locals()[str(county_list[i])+'yield_ssp245_s'] = np.row_stack((np.split(yield_all_model_hist_ssp245_s, 16)[i], np.split(yield_all_model_future_ssp245_s, 16)[i]))
for i in range(0,16):
    locals()[str(county_list[i])+'county_yield_change_2020_ssp585'] = np.zeros((8000))
    locals()[str(county_list[i])+'county_yield_change_2020_ssp585'][:] = np.mean(reference_yield_2001_2020_ssp585[:,:,i,:],axis=2).reshape(8000)
    locals()[str(county_list[i])+'county_yield_change_2099_ssp585'] = np.zeros((8000,2))
    locals()[str(county_list[i])+'county_yield_change_2099_ssp585'][:,0] = 100 * ((np.nanmean(locals()[str(county_list[i])+'yield_ssp585_s'][100:120,:], axis=0))-locals()[str(county_list[i])+'county_yield_change_2020_ssp585'][:])/locals()[str(county_list[i])+'county_yield_change_2020_ssp585'][:]
    locals()[str(county_list[i])+'county_yield_change_2099_ssp585'][:,1] = ((np.nanmean(locals()[str(county_list[i])+'yield_ssp585_s'][100:120,:], axis=0))-locals()[str(county_list[i])+'county_yield_change_2020_ssp585'][:])
    locals()[str(county_list[i])+'county_yield_change_2020_ssp245'] = np.zeros((8000))
    locals()[str(county_list[i])+'county_yield_change_2020_ssp245'][:] = np.mean(reference_yield_2001_2020_ssp245[:,:,i,:],axis=2).reshape(8000)
    locals()[str(county_list[i])+'county_yield_change_2099_ssp245'] = np.zeros((8000,2))
    locals()[str(county_list[i])+'county_yield_change_2099_ssp245'][:,0] = 100 * ((np.nanmean(locals()[str(county_list[i])+'yield_ssp245_s'][100:120,:], axis=0))-locals()[str(county_list[i])+'county_yield_change_2020_ssp245'][:])/locals()[str(county_list[i])+'county_yield_change_2020_ssp245'][:]
    locals()[str(county_list[i])+'county_yield_change_2099_ssp245'][:,1] = ((np.nanmean(locals()[str(county_list[i])+'yield_ssp245_s'][100:120,:], axis=0))-locals()[str(county_list[i])+'county_yield_change_2020_ssp245'][:])
    
    locals()[str(county_list[i])+'county_con_tech_change_2099'] = np.zeros((1000,2))
    locals()[str(county_list[i])+'county_con_tech_change_2099'][:,0] = 100 * (np.mean(future_tech_trend_county_ssp245_con[100:120,i,:], axis=0) - np.mean(locals()[str(county_list[i])+'county_yield_change_2020_ssp245'][:].reshape(8,1000), axis=0)) / np.mean(locals()[str(county_list[i])+'county_yield_change_2020_ssp245'][:].reshape(8,1000), axis=0)
    locals()[str(county_list[i])+'county_con_tech_change_2099'][:,1] = (np.mean(future_tech_trend_county_ssp245_con[100:120,i,:], axis=0) - np.mean(locals()[str(county_list[i])+'county_yield_change_2020_ssp245'][:].reshape(8,1000), axis=0))
    locals()[str(county_list[i])+'county_int_tech_change_2099'] = np.zeros((1000,2))
    locals()[str(county_list[i])+'county_int_tech_change_2099'][:,0] = 100 * (np.mean(future_tech_trend_county_ssp245_int[100:120,i,:], axis=0) - np.mean(locals()[str(county_list[i])+'county_yield_change_2020_ssp245'][:].reshape(8,1000), axis=0)) / np.mean(locals()[str(county_list[i])+'county_yield_change_2020_ssp245'][:].reshape(8,1000), axis=0)
    locals()[str(county_list[i])+'county_int_tech_change_2099'][:,1] = (np.mean(future_tech_trend_county_ssp245_int[100:120,i,:], axis=0) - np.mean(locals()[str(county_list[i])+'county_yield_change_2020_ssp245'][:].reshape(8,1000), axis=0))

    locals()[str(county_list[i])+'county_tech_change_hist'] = np.zeros((1000,2))
    locals()[str(county_list[i])+'county_tech_change_hist'][:,0] = 100 * (np.mean(future_tech_trend_county_ssp245_con[20:40,i,:], axis=0) - np.mean(future_tech_trend_county_ssp245_con[0:20,i,:], axis=0)) / np.mean(future_tech_trend_county_ssp245_con[20:40,i,:], axis=0)
    locals()[str(county_list[i])+'county_tech_change_hist'][:,1] = (np.mean(future_tech_trend_county_ssp245_con[20:40,i,:], axis=0) - np.mean(future_tech_trend_county_ssp245_con[0:20,i,:], axis=0)) 


 
median_yield_change_2099_percent_ssp585 = np.zeros((16))
median_yield_change_2099_percent_ssp245 = np.zeros((16))
median_yield_change_hist_percent = np.zeros((16))
median_yield_change_con_tech_2099_percent = np.zeros((16))
median_yield_change_int_tech_2099_percent = np.zeros((16))
median_yield_change_tech_hist_percent = np.zeros((16))

for i in range(0,16):
    median_yield_change_2099_percent_ssp585[i] = np.median(np.mean((locals()[str(county_list[i])+'county_yield_change_2099_ssp585'][:,0].reshape(8,1000)), axis = 0))
    median_yield_change_2099_percent_ssp245[i] = np.median(np.mean((locals()[str(county_list[i])+'county_yield_change_2099_ssp245'][:,0].reshape(8,1000)), axis = 0))
    median_yield_change_hist_percent[i] = climate_change_only_average_county_yield_hist_change_percent[i]
    median_yield_change_con_tech_2099_percent[i] = np.median(locals()[str(county_list[i])+'county_con_tech_change_2099'][:,0]) 
    median_yield_change_int_tech_2099_percent[i] = np.median(locals()[str(county_list[i])+'county_int_tech_change_2099'][:,0]) 
    median_yield_change_tech_hist_percent[i] = np.median(locals()[str(county_list[i])+'county_tech_change_hist'][:,0])



yield_change_for_shp_85_2099_percent = np.zeros((58))
yield_change_for_shp_85_2099_percent[:] = np.nan
yield_change_for_shp_45_2099_percent = np.zeros((58))
yield_change_for_shp_45_2099_percent[:] = np.nan
yield_change_for_shp_hist_percent = np.zeros((58))
yield_change_for_shp_hist_percent[:] = np.nan

con_tech_change_for_shp_2099_percent = np.zeros((58))
con_tech_change_for_shp_2099_percent[:] = np.nan
int_tech_change_for_shp_2099_percent = np.zeros((58))
int_tech_change_for_shp_2099_percent[:] = np.nan
tech_change_for_shp_hist_percent = np.zeros((58))
tech_change_for_shp_hist_percent[:] = np.nan

county_order_N_S = ['Tehama', 'Butte', 'Glenn', 'Yuba', 'Sutter', 'Colusa', 'Yolo', 'Solano', 'San Joaquin', 'Stanislaus', 'Madera', 'Merced', 'Fresno', 'Tulare', 'Kings', 'Kern']

N_S_order = np.zeros((16))
ca = geopandas.read_file(shp_path+'CA_Counties_TIGER2016.shp')
ca_county_remove_shp = geopandas.read_file(shp_path+'CA_Counties_TIGER2016.shp')
ca_county_remove = ['Sierra', 'Sacramento', 'Santa Barbara', 'Calaveras', 'Ventura','Los Angeles', 'Sonoma', 'San Diego', 'Placer', 'San Francisco', 'Marin', 'Mariposa', 'Lassen', 'Napa',
                    'Shasta', 'Monterey','Trinity', 'Mendocino', 'Inyo', 'Mono', 'Tuolumne', 'San Bernardino', 'Contra Costa', 'Alpine', 'El Dorado', 'San Benito', 'Humboldt','Riverside',
                    'Del Norte', 'Modoc', 'Santa Clara', 'Alameda', 'Nevada', 'Orange', 'Imperial', 'Amador', 'Lake', 'Plumas', 'San Mateo', 'Siskiyou', 'Santa Cruz','San Luis Obispo']
for i in range(0,len(ca_county_remove)):
    ca_county_remove_shp.drop(ca_county_remove_shp.index[ca_county_remove_shp['NAME']==ca_county_remove[i]], inplace=True)

for i in range(0,58):
    for index in range(0,16):
        if county_list[index] == ca.NAME[i]:
            yield_change_for_shp_85_2099_percent[i] = median_yield_change_2099_percent_ssp585[index]
            yield_change_for_shp_45_2099_percent[i] = median_yield_change_2099_percent_ssp245[index]
            yield_change_for_shp_hist_percent[i] = median_yield_change_hist_percent[index]
            con_tech_change_for_shp_2099_percent[i] = median_yield_change_con_tech_2099_percent[index]
            int_tech_change_for_shp_2099_percent[i] = median_yield_change_int_tech_2099_percent[index]
            tech_change_for_shp_hist_percent[i] = median_yield_change_tech_hist_percent[index]

for i in range(0,16):
    N_S_order[np.array(np.where(ca_county_remove_shp['NAME'] == county_order_N_S[i]))] = i+1
ca_county_remove_shp['N_S_order'] = N_S_order.astype(int)

yield_change_for_shp_85_2099_df = pd.DataFrame({'NAME' : ca.NAME, 'ssp585_2099_percent' : yield_change_for_shp_85_2099_percent})
ca_merge_ssp585_2099 =  ca.merge(yield_change_for_shp_85_2099_df, on = 'NAME')
yield_change_for_shp_45_2099_df = pd.DataFrame({'NAME' : ca.NAME, 'ssp245_2099_percent' : yield_change_for_shp_45_2099_percent})
ca_merge_ssp245_2099 =  ca.merge(yield_change_for_shp_45_2099_df, on = 'NAME')
yield_change_for_shp_hist_percent_df = pd.DataFrame({'NAME' : ca.NAME, 'hist_change_percent' : yield_change_for_shp_hist_percent})
ca_merge_hist_change = ca.merge(yield_change_for_shp_hist_percent_df, on = 'NAME')

con_tech_change_for_shp_2099_df = pd.DataFrame({'NAME' : ca.NAME, 'tech_2099_percent' : con_tech_change_for_shp_2099_percent})
ca_merge_con_tech_2099 =  ca.merge(con_tech_change_for_shp_2099_df, on = 'NAME')
int_tech_change_for_shp_2099_df = pd.DataFrame({'NAME' : ca.NAME, 'tech_2099_percent' : int_tech_change_for_shp_2099_percent})
ca_merge_int_tech_2099 =  ca.merge(int_tech_change_for_shp_2099_df, on = 'NAME')
tech_change_for_shp_hist_df = pd.DataFrame({'NAME' : ca.NAME, 'tech_hist_percent' : tech_change_for_shp_hist_percent})
ca_merge_tech_hist =  ca.merge(tech_change_for_shp_hist_df, on = 'NAME')


df_county_yield_ssp585_change = pd.DataFrame()
df_county_yield_ssp245_change = pd.DataFrame()
df_county_yield_hist_change = pd.DataFrame()

df_county_con_tech_2099_change = pd.DataFrame()
df_county_int_tech_2099_change = pd.DataFrame()
df_county_tech_hist_change = pd.DataFrame()

for i in range(0,16):
    df_county_yield_ssp585_change_ind = pd.DataFrame({'County' : str(county_list[i]) , 'Yield Change by 2099' : locals()[str(county_list[i])+'county_yield_change_2099_ssp585'][:,1]})
    df_county_yield_ssp585_change = pd.concat((df_county_yield_ssp585_change, df_county_yield_ssp585_change_ind))
    df_county_yield_ssp245_change_ind = pd.DataFrame({'County' : str(county_list[i]) , 'Yield Change by 2099' : locals()[str(county_list[i])+'county_yield_change_2099_ssp245'][:,1]})
    df_county_yield_ssp245_change = pd.concat((df_county_yield_ssp245_change, df_county_yield_ssp245_change_ind))
    df_county_yield_hist_change_ind = pd.DataFrame({'County' : str(county_list[i]) , 'Hist yield change' : climate_change_only_average_county_yield_hist_change_value[i]})
    df_county_yield_hist_change = pd.concat((df_county_yield_hist_change, df_county_yield_hist_change_ind))
    
    df_county_con_tech_2099_change_ind = pd.DataFrame({'County' : str(county_list[i]) , 'tech Change by 2099' : locals()[str(county_list[i])+'county_con_tech_change_2099'][:,1]})
    df_county_con_tech_2099_change = pd.concat((df_county_con_tech_2099_change , df_county_con_tech_2099_change_ind))
    df_county_int_tech_2099_change_ind = pd.DataFrame({'County' : str(county_list[i]) , 'tech Change by 2099' : locals()[str(county_list[i])+'county_int_tech_change_2099'][:,1]})
    df_county_int_tech_2099_change = pd.concat((df_county_int_tech_2099_change , df_county_int_tech_2099_change_ind))
    df_county_tech_hist_change_ind = pd.DataFrame({'County' : str(county_list[i]) , 'Hist tech Change' : locals()[str(county_list[i])+'county_tech_change_hist'][:,1]})
    df_county_tech_hist_change = pd.concat((df_county_tech_hist_change, df_county_tech_hist_change_ind))

yield_for_shp_obs_2020 = np.zeros((58))
yield_for_shp_obs_2020[:] = np.nan
for i in range(0,58):
    for index in range(0,16):
        if county_list[index] == ca.NAME[i]:
            yield_for_shp_obs_2020[i] = np.median(np.mean(np.mean(reference_yield_2001_2020_ssp585[:,:,index,:],axis=2),axis=0))#+np.median(np.mean(np.mean(reference_yield_2001_2020_ssp245[:,:,index,:],axis=2),axis=0)))/2
yield_for_shp_obs_2020_df = pd.DataFrame({'NAME' : ca.NAME, 'Observation' : yield_for_shp_obs_2020})
ca_merge_obs = ca.merge(yield_for_shp_obs_2020_df)

fig = plt.figure()
fig.set_figheight(45)
fig.set_figwidth(40)
spec = gridspec.GridSpec(nrows=165, ncols=100,hspace = 20)
ax0 = plt.subplot(spec[25:95,0:30])
ca_merge_obs.plot(ax = ax0, column = ca_merge_obs.Observation,edgecolor='black',missing_kwds={'color': 'grey'}, legend = True, cmap = 'Greens',legend_kwds={'orientation': "horizontal", 'aspect' : 10, 'ticks' : [0.5,1,1.5]}, vmin = 0.5, vmax = 1.5)
#ax0.text(-14000000,5150000,'a', fontsize = 35, fontweight='bold')
ax0.set_axis_off()
ax0.set_title('Simulated historical yield over 2001-2020', fontsize = 35, y = 1.08)
ca_county_remove_shp['coords'] = ca_county_remove_shp['geometry'].apply(lambda x: x.representative_point().coords[:])
ca_county_remove_shp['coords'] = [coords[0] for coords in ca_county_remove_shp['coords']]
county_order_N_S = ['Tehama', 'Butte', 'Glenn', 'Yuba', 'Sutter', 'Colusa', 'Yolo', 'Solano', 'San Joaquin', 'Stanislaus', 'Madera', 'Merced', 'Fresno', 'Tulare', 'Kings', 'Kern']
ticks_county_order_N_S = ['[1]', '[2]', '[3]', '[4]', '[5]', '[6]', '[7]', '[8]', '[9]', '[10]', '[11]', '[12]', '[13]', '[14]', '[15]', '[16]']
idx_county_order_N_S = ['[1] Tehama', '[2] Butte', '[3] Glenn', '[4] Yuba', '[5] Sutter', '[6] Colusa', '[7] Yolo', '[8] Solano', '[9] San Joaquin', '[10] Stanislaus', '[11] Madera', '[12] Merced', '[13] Fresno', '[14] Tulare', '[15] Kings', '[16] Kern']
for idx in range(8):
    ax0.text(x = -13800000, y = (3100000-100000*idx), s=idx_county_order_N_S[idx], fontsize =35)
for idx in range(8,16):
    ax0.text(x = -13200000, y = (3100000-100000*(idx-8)), s=idx_county_order_N_S[idx], fontsize =35)

ax1 = plt.subplot(spec[5:55,45:65])
ca_merge_tech_hist.plot(ax = ax1, column = ca_merge_tech_hist.tech_hist_percent, edgecolor='black',missing_kwds={'color': 'grey'}, legend = True, cmap = 'Purples',legend_kwds={'orientation': "horizontal", 'ticks': [0, 15,30]}, vmin = 0, vmax = 30)
#ax1.text(-14700000,5370000,'b', fontsize = 35, fontweight='bold')
fig = ax1.figure
cb_ax = fig.axes[3]
cb_ax.tick_params(labelsize = 25)
#cb_ax.set_title("Yield change, %", fontsize=35, y=-2.5)
cb_ax.set_position([0.421, 0.267, 0.17, 0.4])
cb_ax.text(s = '%', x = 30.7, y = 5 ,fontsize = 25)
cb_ax = fig.axes[1]
cb_ax.tick_params(labelsize = 35)
cb_ax.set_position([0.15, 0.1, 0.2, 0.4])
cb_ax.text(x = 0.88, y = -1, s = 'ton/acre', fontsize = 35)
ax1.set_axis_off()

ax1_box = plt.subplot(spec[5:40,35:45])
norm = matplotlib.colors.Normalize(vmax = 30,vmin = 0)
my_pal = {'Butte' : cm.Purples(norm(median_yield_change_tech_hist_percent[0])), 'Colusa': cm.Purples(norm(median_yield_change_tech_hist_percent[1])), 'Fresno' : cm.Purples(norm(median_yield_change_tech_hist_percent[2])), 'Glenn' : cm.Purples(norm(median_yield_change_tech_hist_percent[3])),
          'Kern' : cm.Purples(norm(median_yield_change_tech_hist_percent[4])), 'Kings' : cm.Purples(norm(median_yield_change_tech_hist_percent[5])), 'Madera' : cm.Purples(norm(median_yield_change_tech_hist_percent[6])), 'Merced' : cm.Purples(norm(median_yield_change_tech_hist_percent[7])),
          'San Joaquin' : cm.Purples(norm(median_yield_change_tech_hist_percent[8])), 'Solano' : cm.Purples(norm(median_yield_change_tech_hist_percent[9])), 'Stanislaus' : cm.Purples(norm(median_yield_change_tech_hist_percent[10])), 'Sutter' : cm.Purples(norm(median_yield_change_tech_hist_percent[11])),
          'Tehama' :cm.Purples(norm(median_yield_change_tech_hist_percent[12])), 'Tulare' : cm.Purples(norm(median_yield_change_tech_hist_percent[13])), 'Yolo' : cm.Purples(norm(median_yield_change_tech_hist_percent[14])), 'Yuba': cm.Purples(norm(median_yield_change_tech_hist_percent[15]))}
sns.barplot(ax = ax1_box, data = df_county_tech_hist_change, x = 'Hist tech Change', y = 'County', order = county_order_N_S ,palette = my_pal,errorbar= ('pi',50), capsize = 0.25)
ax1_box.text(0.55, -1.5, 'Historical', fontsize = 30)
ax1_box.text(1.2,-3, 'Yield change from innovation', fontsize = 35)
ax1_box.set_xticks([0, 0.2, 0.4, 0.6])
ax1_box.set_xticklabels([0, 0.2, 0.4, 0.6])
plt.xlabel('ton/acre', fontsize = 25)
plt.ylabel('')
plt.xticks(fontsize = 25)
plt.yticks(np.arange(0,16), ticks_county_order_N_S, fontsize = 25)
#plt.xlim(-50,300)
#plt.axvline(x=-100, linestyle = 'dashed', color = 'r')
#plt.axvline(x=0, linestyle = 'dashed', color = 'r')


ax2 = plt.subplot(spec[5:55,65:85])
ca_merge_con_tech_2099.plot(ax = ax2, column = ca_merge_con_tech_2099.tech_2099_percent, edgecolor='black',missing_kwds={'color': 'grey'}, legend = True, cmap = 'Purples', vmin = 0, vmax = 250,legend_kwds={'orientation': "horizontal", 'ticks': [0, 125,250]})
ax2.set_axis_off()
cb_ax = fig.axes[6]
cb_ax.set_position([0.706, 0.267, 0.17, 0.4])
cb_ax.text(s = '%', x = 256, y = 40 ,fontsize = 25)
cb_ax.tick_params(labelsize = 25)

ax2_box = plt.subplot(spec[5:40,90:100])
norm = matplotlib.colors.Normalize(vmax = 250,vmin = 0)

my_pal = {'Butte' : cm.Purples(norm(median_yield_change_con_tech_2099_percent[0])), 'Colusa': cm.Purples(norm(median_yield_change_con_tech_2099_percent[1])), 'Fresno' : cm.Purples(norm(median_yield_change_con_tech_2099_percent[2])), 'Glenn' : cm.Purples(norm(median_yield_change_con_tech_2099_percent[3])),
          'Kern' : cm.Purples(norm(median_yield_change_con_tech_2099_percent[4])), 'Kings' : cm.Purples(norm(median_yield_change_con_tech_2099_percent[5])), 'Madera' : cm.Purples(norm(median_yield_change_con_tech_2099_percent[6])), 'Merced' : cm.Purples(norm(median_yield_change_con_tech_2099_percent[7])),
          'San Joaquin' : cm.Purples(norm(median_yield_change_con_tech_2099_percent[8])), 'Solano' : cm.Purples(norm(median_yield_change_con_tech_2099_percent[9])), 'Stanislaus' : cm.Purples(norm(median_yield_change_con_tech_2099_percent[10])), 'Sutter' : cm.Purples(norm(median_yield_change_con_tech_2099_percent[11])),
          'Tehama' :cm.Purples(norm(median_yield_change_con_tech_2099_percent[12])), 'Tulare' : cm.Purples(norm(median_yield_change_con_tech_2099_percent[13])), 'Yolo' : cm.Purples(norm(median_yield_change_con_tech_2099_percent[14])), 'Yuba': cm.Purples(norm(median_yield_change_con_tech_2099_percent[15]))}
sns.barplot(ax = ax2_box, data = df_county_con_tech_2099_change, x = 'tech Change by 2099', y = 'County', order = county_order_N_S ,palette = my_pal,errorbar= ('pi',50), capsize = 0.25)
ax2_box.text(-1.9, -1.5, 'Future', fontsize = 30)
#ax1.text(-10500000,5370000,'c', fontsize = 35, fontweight='bold')
ax2_box.set_xticks([0, 1, 2,3])
ax2_box.set_xticklabels([0, 1, 2, 3])
#ax2_box.text(-450, -1, 'High-tech', fontsize = 30)
plt.xlabel('ton/acre', fontsize = 25)
plt.ylabel('')
plt.xticks(fontsize = 25)
plt.yticks(np.arange(0,16), ticks_county_order_N_S, fontsize = 25)
#plt.xlim(-50,300)
#plt.axvline(x=-100, linestyle = 'dashed', color = 'r')
#plt.axvline(x=0, linestyle = 'dashed', color = 'r')
#ax2_box.spines['top'].set_visible(False)
#ax2_box.spines['right'].set_visible(False)

ax3 = plt.subplot(spec[63:113,45:65])
ca_merge_ssp585_2099.plot(ax = ax3, column = ca_merge_hist_change.hist_change_percent,edgecolor='black',missing_kwds={'color': 'grey'}, legend = True, cmap = 'OrRd_r', figsize = (15,15),legend_kwds={'orientation': "horizontal", 'ticks' : [-20,-10,0]}, vmin = -20, vmax=0)
cb_ax = fig.axes[9]
cb_ax.tick_params(labelsize = 25)
cb_ax.set_position([0.421, 0.100, 0.17, 0.3])
cb_ax.text(s = '%', x = 0.15, y = -5 ,fontsize = 25)
ax3.set_axis_off()

ax3_box = plt.subplot(spec[63:97,35:45])
norm = matplotlib.colors.Normalize(vmax = 0,vmin = -20)

my_pal = {'Butte' : cm.OrRd_r(norm(median_yield_change_hist_percent[0])), 'Colusa': cm.OrRd_r(norm(median_yield_change_hist_percent[1])), 'Fresno' : cm.OrRd_r(norm(median_yield_change_hist_percent[2])), 'Glenn' : cm.OrRd_r(norm(median_yield_change_hist_percent[3])),
          'Kern' : cm.OrRd_r(norm(median_yield_change_hist_percent[4])), 'Kings' : cm.OrRd_r(norm(median_yield_change_hist_percent[5])), 'Madera' : cm.OrRd_r(norm(median_yield_change_hist_percent[6])), 'Merced' : cm.OrRd_r(norm(median_yield_change_hist_percent[7])),
          'San Joaquin' : cm.OrRd_r(norm(median_yield_change_hist_percent[8])), 'Solano' : cm.OrRd_r(norm(median_yield_change_hist_percent[9])), 'Stanislaus' : cm.OrRd_r(norm(median_yield_change_hist_percent[10])), 'Sutter' : cm.OrRd_r(norm(median_yield_change_hist_percent[11])),
          'Tehama' :cm.OrRd_r(norm(median_yield_change_hist_percent[12])), 'Tulare' : cm.OrRd_r(norm(median_yield_change_hist_percent[13])), 'Yolo' : cm.OrRd_r(norm(median_yield_change_hist_percent[14])), 'Yuba': cm.OrRd_r(norm(median_yield_change_hist_percent[15]))}

sns.barplot(ax = ax3_box, data = df_county_yield_hist_change, x = 'Hist yield change', y = 'County', order = county_order_N_S ,palette = my_pal,errorbar= ('pi', 50), capsize = 0.25)
for line in ax3_box.lines:
    line.set_linewidth(2)
#ax3.text(-14700000,5370000,'d', fontsize = 35, fontweight='bold')

ax3_box.text(0.075, -4, 'Yield change from climate change', fontsize = 35)
ax3_box.text(-0.005, -1.5, 'Historical LOCAv2', fontsize = 30)

#ax3_box.text(20, -1, 'RCP4.5', fontsize = 30)

ax3_box.set_xticks([-0.06, -0.03, 0])
ax3_box.set_xticklabels(['-0.06', '-0.03', '0'])
plt.xlabel('ton/acre', fontsize = 25)
plt.ylabel('')
plt.xticks(fontsize = 25)
plt.yticks(np.arange(0,16), ticks_county_order_N_S, fontsize = 25)
#plt.xlim(-105,50)
#plt.axvline(x=-100, linestyle = 'dashed', color = 'r')
#plt.axvline(x=0, linestyle = 'dashed', color = 'r')
#ax3_box.spines['top'].set_visible(False)
#ax3_box.spines['right'].set_visible(False)


ax4 = plt.subplot(spec[63:113,65:85])
ca_merge_ssp585_2099.plot(ax = ax4,column = ca_merge_ssp585_2099.ssp585_2099_percent, edgecolor='black',missing_kwds={'color': 'grey'},legend = True,legend_kwds={'orientation': "horizontal", 'ticks' : [-70, -35,0]}, cmap = 'OrRd_r', figsize = (15,15),vmin = -70, vmax = 0)
cb_ax = fig.axes[12]
cb_ax.tick_params(labelsize = 25)
cb_ax.set_position([0.707,0.100, 0.17, 0.3])
cb_ax.text(s = '%', x = 2, y = -60 ,fontsize = 25)
ax4.set_axis_off()
ax4_box = plt.subplot(spec[63:97,90:100])
norm = matplotlib.colors.Normalize(vmax = 0,vmin = -70)

my_pal = {'Butte' : cm.OrRd_r(norm(median_yield_change_2099_percent_ssp585[0])), 'Colusa': cm.OrRd_r(norm(median_yield_change_2099_percent_ssp585[1])), 'Fresno' : cm.OrRd_r(norm(median_yield_change_2099_percent_ssp585[2])), 'Glenn' : cm.OrRd_r(norm(median_yield_change_2099_percent_ssp585[3])),
          'Kern' : cm.OrRd_r(norm(median_yield_change_2099_percent_ssp585[4])), 'Kings' : cm.OrRd_r(norm(median_yield_change_2099_percent_ssp585[5])), 'Madera' : cm.OrRd_r(norm(median_yield_change_2099_percent_ssp585[6])), 'Merced' : cm.OrRd_r(norm(median_yield_change_2099_percent_ssp585[7])),
          'San Joaquin' : cm.OrRd_r(norm(median_yield_change_2099_percent_ssp585[8])), 'Solano' : cm.OrRd_r(norm(median_yield_change_2099_percent_ssp585[9])), 'Stanislaus' : cm.OrRd_r(norm(median_yield_change_2099_percent_ssp585[10])), 'Sutter' : cm.OrRd_r(norm(median_yield_change_2099_percent_ssp585[11])),
          'Tehama' :cm.OrRd_r(norm(median_yield_change_2099_percent_ssp585[12])), 'Tulare' : cm.OrRd_r(norm(median_yield_change_2099_percent_ssp585[13])), 'Yolo' : cm.OrRd_r(norm(median_yield_change_2099_percent_ssp585[14])), 'Yuba': cm.OrRd_r(norm(median_yield_change_2099_percent_ssp585[15]))}
sns.barplot(ax = ax4_box, data = df_county_yield_ssp585_change, x = 'Yield Change by 2099', y = 'County', order = county_order_N_S ,palette = my_pal,errorbar= ('pi',50), capsize = 0.25)
for line in ax4_box.lines:
    line.set_linewidth(2)
#plt.xlim(-0.4,0)
#ax3.text(-10500000,5370000,'e', fontsize = 35, fontweight='bold')
#ax4_box.text(-260, -1, 'RCP8.5', fontsize = 30)
ax4_box.text(-2.1 , -1.5, 'Future LOCAv2-SSP585', fontsize = 30)
ax4_box.set_xticks([-0.9,-0.6,-0.3,0])
ax4_box.set_xticklabels([-0.9,-0.6,-0.3,0])
plt.xlabel('ton/acre', fontsize = 25)
plt.ylabel('')
plt.xticks(fontsize = 25)
plt.yticks(np.arange(0,16), ticks_county_order_N_S, fontsize = 25)
#plt.xlim(-105,50)
#plt.axvline(x=-100, linestyle = 'dashed', color = 'r')
#plt.axvline(x=0, linestyle = 'dashed', color = 'r')
#ax4_box.spines['top'].set_visible(False)
#ax4_box.spines['right'].set_visible(False)

ax5 = plt.subplot(spec[115:165,45:65])
ca_merge_rcp85_2099.plot(ax = ax5, column = ca_merge_hist_change_rcp85.hist_change_percent,edgecolor='black',missing_kwds={'color': 'grey'}, legend = True, cmap = 'OrRd_r', figsize = (15,15),legend_kwds={'orientation': "horizontal", 'ticks' : [-6,-3,0]}, vmin = -6, vmax=0)
cb_ax = fig.axes[15]
cb_ax.tick_params(labelsize = 25)
cb_ax.set_position([0.421, -0.15, 0.17, 0.3])
cb_ax.text(s = '%', x = 0.15, y = -5 ,fontsize = 25)
ax5.set_axis_off()

ax5_box = plt.subplot(spec[115:150,35:45])
norm = matplotlib.colors.Normalize(vmax = 0,vmin = -6)

my_pal = {'Butte' : cm.OrRd_r(norm(median_yield_change_hist_percent_rcp85[0])), 'Colusa': cm.OrRd_r(norm(median_yield_change_hist_percent_rcp85[1])), 'Fresno' : cm.OrRd_r(norm(median_yield_change_hist_percent_rcp85[2])), 'Glenn' : cm.OrRd_r(norm(median_yield_change_hist_percent_rcp85[3])),
          'Kern' : cm.OrRd_r(norm(median_yield_change_hist_percent_rcp85[4])), 'Kings' : cm.OrRd_r(norm(median_yield_change_hist_percent_rcp85[5])), 'Madera' : cm.OrRd_r(norm(median_yield_change_hist_percent_rcp85[6])), 'Merced' : cm.OrRd_r(norm(median_yield_change_hist_percent_rcp85[7])),
          'San Joaquin' : cm.OrRd_r(norm(median_yield_change_hist_percent_rcp85[8])), 'Solano' : cm.OrRd_r(norm(median_yield_change_hist_percent_rcp85[9])), 'Stanislaus' : cm.OrRd_r(norm(median_yield_change_hist_percent_rcp85[10])), 'Sutter' : cm.OrRd_r(norm(median_yield_change_hist_percent_rcp85[11])),
          'Tehama' :cm.OrRd_r(norm(median_yield_change_hist_percent_rcp85[12])), 'Tulare' : cm.OrRd_r(norm(median_yield_change_hist_percent_rcp85[13])), 'Yolo' : cm.OrRd_r(norm(median_yield_change_hist_percent_rcp85[14])), 'Yuba': cm.OrRd_r(norm(median_yield_change_hist_percent_rcp85[15]))}

sns.barplot(ax = ax5_box, data = df_county_yield_hist_change_rcp85, x = 'Hist yield change', y = 'County', order = county_order_N_S ,palette = my_pal,errorbar= ('pi', 50), capsize = 0.25)
for line in ax3_box.lines:
    line.set_linewidth(2)
#ax3.text(-14700000,5370000,'d', fontsize = 35, fontweight='bold')

#ax5_box.text(0.075, -3, 'Yield change from climate change', fontsize = 35)
ax5_box.text(0.01, -1.5, 'Historical MACA', fontsize = 30)

#ax3_box.text(20, -1, 'RCP4.5', fontsize = 30)

ax5_box.set_xticks([-0.06, -0.03, 0])
ax5_box.set_xticklabels(['-0.06', '-0.03', '0'])
plt.xlabel('ton/acre', fontsize = 25)
plt.ylabel('')
plt.xticks(fontsize = 25)
plt.yticks(np.arange(0,16), ticks_county_order_N_S, fontsize = 25)

ax6 = plt.subplot(spec[115:165,65:85])
ca_merge_rcp85_2099.plot(ax = ax6,column = ca_merge_rcp85_2099.rcp85_2099_percent, edgecolor='black',missing_kwds={'color': 'grey'},legend = True,legend_kwds={'orientation': "horizontal", 'ticks' : [-70, -35,0]}, cmap = 'OrRd_r', figsize = (15,15),vmin = -70, vmax = 0)
cb_ax = fig.axes[18]
cb_ax.tick_params(labelsize = 25)
cb_ax.set_position([0.707,-0.15, 0.17, 0.3])
cb_ax.text(s = '%', x = 2, y = -60 ,fontsize = 25)
ax6.set_axis_off()
ax6_box = plt.subplot(spec[115:150,90:100])
norm = matplotlib.colors.Normalize(vmax = 0,vmin = -70)

my_pal = {'Butte' : cm.OrRd_r(norm(median_yield_change_2099_percent_rcp85[0])), 'Colusa': cm.OrRd_r(norm(median_yield_change_2099_percent_rcp85[1])), 'Fresno' : cm.OrRd_r(norm(median_yield_change_2099_percent_rcp85[2])), 'Glenn' : cm.OrRd_r(norm(median_yield_change_2099_percent_rcp85[3])),
          'Kern' : cm.OrRd_r(norm(median_yield_change_2099_percent_rcp85[4])), 'Kings' : cm.OrRd_r(norm(median_yield_change_2099_percent_rcp85[5])), 'Madera' : cm.OrRd_r(norm(median_yield_change_2099_percent_rcp85[6])), 'Merced' : cm.OrRd_r(norm(median_yield_change_2099_percent_rcp85[7])),
          'San Joaquin' : cm.OrRd_r(norm(median_yield_change_2099_percent_rcp85[8])), 'Solano' : cm.OrRd_r(norm(median_yield_change_2099_percent_rcp85[9])), 'Stanislaus' : cm.OrRd_r(norm(median_yield_change_2099_percent_rcp85[10])), 'Sutter' : cm.OrRd_r(norm(median_yield_change_2099_percent_rcp85[11])),
          'Tehama' :cm.OrRd_r(norm(median_yield_change_2099_percent_rcp85[12])), 'Tulare' : cm.OrRd_r(norm(median_yield_change_2099_percent_rcp85[13])), 'Yolo' : cm.OrRd_r(norm(median_yield_change_2099_percent_rcp85[14])), 'Yuba': cm.OrRd_r(norm(median_yield_change_2099_percent_rcp85[15]))}
sns.barplot(ax = ax6_box, data = df_county_yield_rcp85_change, x = 'Yield Change by 2099', y = 'County', order = county_order_N_S ,palette = my_pal,errorbar= ('pi',50), capsize = 0.25)
for line in ax4_box.lines:
    line.set_linewidth(2)
#plt.xlim(-0.4,0)
#ax3.text(-10500000,5370000,'e', fontsize = 35, fontweight='bold')
#ax4_box.text(-260, -1, 'RCP8.5', fontsize = 30)
ax6_box.text(-1.9 , -1.5, 'Future MACA-RCP8.5', fontsize = 30)
ax6_box.set_xticks([-0.9,-0.6,-0.3,0])
ax6_box.set_xticklabels([-0.9,-0.6,-0.3,0])
plt.xlabel('ton/acre', fontsize = 25)
plt.ylabel('')
plt.xticks(fontsize = 25)
plt.yticks(np.arange(0,16), ticks_county_order_N_S, fontsize = 25)
plt.savefig(save_path+'map.pdf', dpi = 300, bbox_inches='tight')


## Figure 5: Uncertainty

def poly_transform(X,Y):
    poly_array = np.zeros((Y.shape[0],Y.shape[1]))
    Y_len = Y.shape[1]
    for i in range(Y_len):
        Poly_fit = np.polyfit(X, Y[:,i], deg = 4)
        Poly_func = np.poly1d(Poly_fit)
        Poly_pred = Poly_func(X)
        poly_array[:,i] = Poly_pred
    return(poly_array)

yield_all_sum_ssp245 = np.row_stack((yield_all_hist_ssp245, yield_all_future_ssp245))
yield_all_sum_ssp245_s = np.row_stack((yield_all_hist_ssp245_s, yield_all_future_ssp245_s))
yield_all_sum_ssp245_m = np.row_stack((yield_all_hist_ssp245_m, yield_all_future_ssp245_m))
yield_all_sum_ssp585 = np.row_stack((yield_all_hist_ssp585, yield_all_future_ssp585))
yield_all_sum_ssp585_s = np.row_stack((yield_all_hist_ssp585_s, yield_all_future_ssp585_s))
yield_all_sum_ssp585_m = np.row_stack((yield_all_hist_ssp585_m, yield_all_future_ssp585_m))



poly_X = np.arange(1980,2100)
yield_all_sum_ssp245_s_poly = poly_transform(poly_X, yield_all_sum_ssp245_s).reshape(120,8,1000)[40:120]
yield_all_sum_ssp585_s_poly = poly_transform(poly_X, yield_all_sum_ssp585_s).reshape(120,8,1000)[40:120]
yield_all_sum_ssp245_m_poly = poly_transform(poly_X, yield_all_sum_ssp245_m).reshape(120,8,1000)[40:120]
yield_all_sum_ssp585_m_poly = poly_transform(poly_X, yield_all_sum_ssp585_m).reshape(120,8,1000)[40:120]
yield_all_sum_ssp245_poly = poly_transform(poly_X, yield_all_sum_ssp245).reshape(120,8,1000)[40:120]
yield_all_sum_ssp585_poly = poly_transform(poly_X, yield_all_sum_ssp585).reshape(120,8,1000)[40:120]

yield_all_sum_ssp245_2020_2099 = yield_all_sum_ssp245[40:120]
yield_all_sum_ssp245_s_2020_2099 =  yield_all_sum_ssp245_s[40:120]
yield_all_sum_ssp245_m_2020_2099 =  yield_all_sum_ssp245_m[40:120]
yield_all_sum_ssp585_2020_2099 =  yield_all_sum_ssp585[40:120]
yield_all_sum_ssp585_s_2020_2099=  yield_all_sum_ssp585_s[40:120]
yield_all_sum_ssp585_m_2020_2099=  yield_all_sum_ssp585_m[40:120]

num_rcp = 2
num_stat_model = 1000
num_tech = 3
num_clim_model = 8
## calculate climate model uncertainty MC
MC_ssp245 = np.zeros(80)
MC_ssp245_s = np.zeros(80)
MC_ssp245_m = np.zeros(80)
MC_ssp585 = np.zeros(80)
MC_ssp585_s = np.zeros(80) 
MC_ssp585_m = np.zeros(80) 
for year in range(0,80):
    MC_ssp245[year] = np.var(np.median(yield_all_sum_ssp245_poly, axis = 2)[year,:])
    MC_ssp245_s[year] = np.var(np.median(yield_all_sum_ssp245_s_poly, axis = 2)[year,:])
    MC_ssp245_m[year] = np.var(np.median(yield_all_sum_ssp245_m_poly, axis = 2)[year,:])
    MC_ssp585[year] = np.var(np.median(yield_all_sum_ssp585_poly, axis = 2)[year,:])
    MC_ssp585_s[year] = np.var(np.median(yield_all_sum_ssp585_s_poly, axis = 2)[year,:])
    MC_ssp585_m[year] = np.var(np.median(yield_all_sum_ssp585_m_poly, axis = 2)[year,:])   
    
MC_time_series = (MC_ssp245+MC_ssp245_s+MC_ssp245_m+MC_ssp585+MC_ssp585_s+MC_ssp585_m)/6

## calculate stat model uncertainty MS
MS_ssp245 = np.zeros(80)
MS_ssp245_s = np.zeros(80)
MS_ssp245_m = np.zeros(80)
MS_ssp585 = np.zeros(80)
MS_ssp585_s = np.zeros(80) 
MS_ssp585_m = np.zeros(80) 
for year in range(0,80):
        MS_ssp245[year] = np.var(np.mean(yield_all_sum_ssp245_poly, axis = 1)[year,:])
        MS_ssp245_s[year] = np.var(np.mean(yield_all_sum_ssp245_s_poly, axis = 1)[year,:])
        MS_ssp245_m[year] = np.var(np.mean(yield_all_sum_ssp245_m_poly, axis = 1)[year,:])
        MS_ssp585[year] = np.var(np.mean(yield_all_sum_ssp585_poly, axis = 1)[year,:])  
        MS_ssp585_s[year] = np.var(np.mean(yield_all_sum_ssp585_s_poly, axis = 1)[year,:])
        MS_ssp585_m[year] = np.var(np.mean(yield_all_sum_ssp585_m_poly, axis = 1)[year,:])
MS_time_series = (MS_ssp245 + MS_ssp245_s + MS_ssp245_m + MS_ssp585 + MS_ssp585_s + MS_ssp585_m)/6
## calculate tech trend scenario uncertainty ST
yield_all_sum_ssp245_tech_scenario = np.zeros((80,8,1000,3))    
yield_all_sum_ssp245_tech_scenario[:,:,:,0] = yield_all_sum_ssp245_poly
yield_all_sum_ssp245_tech_scenario[:,:,:,1] = yield_all_sum_ssp245_s_poly
yield_all_sum_ssp245_tech_scenario[:,:,:,2] = yield_all_sum_ssp245_m_poly
yield_all_sum_ssp585_tech_scenario = np.zeros((80,8,1000,3))    
yield_all_sum_ssp585_tech_scenario[:,:,:,0] = yield_all_sum_ssp585_poly
yield_all_sum_ssp585_tech_scenario[:,:,:,1] = yield_all_sum_ssp585_s_poly
yield_all_sum_ssp585_tech_scenario[:,:,:,2] = yield_all_sum_ssp585_m_poly

ST_ssp245 = np.zeros(80)
ST_ssp585 = np.zeros(80)
for year in range(0,80):
    ST_ssp245[year] = np.var(np.median(np.mean(yield_all_sum_ssp245_tech_scenario, axis = 1), axis = 1)[year,:])
    ST_ssp585[year] = np.var(np.median(np.mean(yield_all_sum_ssp585_tech_scenario, axis = 1), axis = 1)[year,:])
ST_time_series = ((ST_ssp245 + ST_ssp585)/2)

## calculate rcp scenario uncertainty SR
yield_all_sum_rcp_scenario_tech = np.zeros((80,8,1000,2))    
yield_all_sum_rcp_scenario_tech[:,:,:,0] = yield_all_sum_ssp245_poly
yield_all_sum_rcp_scenario_tech[:,:,:,1] = yield_all_sum_ssp585_poly
yield_all_sum_rcp_scenario_no_tech = np.zeros((80,8,1000,2))    
yield_all_sum_rcp_scenario_no_tech[:,:,:,0] = yield_all_sum_ssp245_s_poly
yield_all_sum_rcp_scenario_no_tech[:,:,:,1] = yield_all_sum_ssp585_s_poly
yield_all_sum_rcp_scenario_int_tech = np.zeros((80,8,1000,2))    
yield_all_sum_rcp_scenario_int_tech[:,:,:,0] = yield_all_sum_ssp245_m_poly
yield_all_sum_rcp_scenario_int_tech[:,:,:,1] = yield_all_sum_ssp585_m_poly

SR_tech = np.zeros(80)
SR_no_tech = np.zeros(80)
SR_int_tech = np.zeros(80)
for year in range(0,80):
    SR_tech[year] = np.var(np.median(np.mean(yield_all_sum_rcp_scenario_tech, axis = 1), axis = 1)[year,:])
    SR_no_tech[year] = np.var(np.median(np.mean(yield_all_sum_rcp_scenario_no_tech, axis = 1), axis = 1)[year,:])
    SR_int_tech[year] = np.var(np.median(np.mean(yield_all_sum_rcp_scenario_int_tech, axis = 1), axis = 1)[year,:])
SR_time_series = ((SR_tech + SR_no_tech + SR_int_tech)/3)
## calculate internal 
Intvar_ssp245 = np.zeros(80)
Intvar_ssp245_s = np.zeros(80)
Intvar_ssp245_m = np.zeros(80)
Intvar_ssp585 = np.zeros(80)
Intvar_ssp585_s = np.zeros(80)
Intvar_ssp585_m = np.zeros(80)
residual_ssp245 = np.zeros((80,8,1000))
residual_ssp245_s = np.zeros((80,8,1000))
residual_ssp245_m = np.zeros((80,8,1000))
residual_ssp585 = np.zeros((80,8,1000))
residual_ssp585_s = np.zeros((80,8,1000))
residual_ssp585_m = np.zeros((80,8,1000))

for trial in range(0,1000):
    for climate in range(0,8):
        residual_ssp245[:,climate,trial] = yield_all_sum_ssp245_poly[:,climate,trial]-np.array(pd.DataFrame(yield_all_sum_ssp245_2020_2099.reshape(80,8,1000)[:,climate,trial]).rolling(10,min_periods=5, center = True).mean()).reshape(80)
        residual_ssp245_s[:,climate,trial] =  yield_all_sum_ssp245_s_poly[:,climate,trial]-np.array(pd.DataFrame(yield_all_sum_ssp245_s_2020_2099.reshape(80,8,1000)[:,climate,trial]).rolling(10,min_periods=5, center = True).mean()).reshape(80)
        residual_ssp245_m[:,climate,trial] =  yield_all_sum_ssp245_m_poly[:,climate,trial]-np.array(pd.DataFrame(yield_all_sum_ssp245_m_2020_2099.reshape(80,8,1000)[:,climate,trial]).rolling(10,min_periods=5, center = True).mean()).reshape(80)
        residual_ssp585[:,climate,trial] =  yield_all_sum_ssp585_poly[:,climate,trial]-np.array(pd.DataFrame(yield_all_sum_ssp585_2020_2099.reshape(80,8,1000)[:,climate,trial]).rolling(10,min_periods=5, center = True).mean()).reshape(80)
        residual_ssp585_s[:,climate,trial] =  yield_all_sum_ssp585_s_poly[:,climate,trial]-np.array(pd.DataFrame(yield_all_sum_ssp585_s_2020_2099.reshape(80,8,1000)[:,climate,trial]).rolling(10,min_periods=5, center = True).mean()).reshape(80)
        residual_ssp585_m[:,climate,trial] =  yield_all_sum_ssp585_m_poly[:,climate,trial]-np.array(pd.DataFrame(yield_all_sum_ssp585_m_2020_2099.reshape(80,8,1000)[:,climate,trial]).rolling(10,min_periods=5, center = True).mean()).reshape(80)


for year in range(0,80):
    Intvar_ssp245[year] = np.var(residual_ssp245[year,:,:])
    Intvar_ssp245_s[year] = np.var(residual_ssp245_s[year,:,:])
    Intvar_ssp245_m[year] = np.var(residual_ssp245_m[year,:,:])
    Intvar_ssp585[year] = np.var(residual_ssp585[year,:,:])
    Intvar_ssp585_s[year] = np.var(residual_ssp585_s[year,:,:])
    Intvar_ssp585_m[year] = np.var(residual_ssp585_m[year,:,:])
Intvar_time_series = ((Intvar_ssp245+Intvar_ssp245_s+Intvar_ssp245_m+Intvar_ssp585+Intvar_ssp585_s+Intvar_ssp585_m)/6)
#####calculate mean/median yield + uncertainty
mean_median_yield_2020_2099_ssp245 = np.median(np.mean(yield_all_sum_ssp245_2020_2099.reshape(80,8,1000),axis=1),axis=1)
mean_median_yield_2020_2099_ssp245_s =  np.median(np.mean(yield_all_sum_ssp245_s_2020_2099.reshape(80,8,1000),axis=1),axis=1)
mean_median_yield_2020_2099_ssp245_m =  np.median(np.mean(yield_all_sum_ssp245_m_2020_2099.reshape(80,8,1000),axis=1),axis=1)
mean_median_yield_2020_2099_ssp585 =  np.median(np.mean(yield_all_sum_ssp585_2020_2099.reshape(80,8,1000),axis=1),axis=1)
mean_median_yield_2020_2099_ssp585_s =  np.median(np.mean(yield_all_sum_ssp585_s_2020_2099.reshape(80,8,1000),axis=1),axis=1)
mean_median_yield_2020_2099_ssp585_m =  np.median(np.mean(yield_all_sum_ssp585_m_2020_2099.reshape(80,8,1000),axis=1),axis=1)

mean_median_yield_2020_2099_ssp245_10_yr_running_mean = pd.DataFrame(mean_median_yield_2020_2099_ssp245).rolling(10,min_periods=5, center = True).mean()
mean_median_yield_2020_2099_ssp245_s_10_yr_running_mean = pd.DataFrame(mean_median_yield_2020_2099_ssp245_s).rolling(10,min_periods=5, center = True).mean()
mean_median_yield_2020_2099_ssp245_m_10_yr_running_mean = pd.DataFrame(mean_median_yield_2020_2099_ssp245_m).rolling(10,min_periods=5, center = True).mean()

mean_median_yield_2020_2099_ssp585_10_yr_running_mean = pd.DataFrame(mean_median_yield_2020_2099_ssp585).rolling(10,min_periods=5, center = True).mean()
mean_median_yield_2020_2099_ssp585_s_10_yr_running_mean = pd.DataFrame(mean_median_yield_2020_2099_ssp585_s).rolling(10,min_periods=5, center = True).mean()
mean_median_yield_2020_2099_ssp585_m_10_yr_running_mean = pd.DataFrame(mean_median_yield_2020_2099_ssp585_m).rolling(10,min_periods=5, center = True).mean()


mean_median_yield_2020_2099 = np.array((mean_median_yield_2020_2099_ssp245_10_yr_running_mean + mean_median_yield_2020_2099_ssp245_s_10_yr_running_mean + mean_median_yield_2020_2099_ssp245_m_10_yr_running_mean + mean_median_yield_2020_2099_ssp585_10_yr_running_mean
                               + mean_median_yield_2020_2099_ssp585_s_10_yr_running_mean + mean_median_yield_2020_2099_ssp585_m_10_yr_running_mean)/6).reshape(80)

y_upper_tech_no_tech = np.zeros((5,80))
y_lower_tech_no_tech = np.zeros((5,80))
y_upper_tech_no_tech[0,:] = mean_median_yield_2020_2099 + (np.sqrt(MC_time_series)+np.sqrt(MS_time_series)+np.sqrt(ST_time_series)+np.sqrt(SR_time_series)+np.sqrt(Intvar_time_series))
y_lower_tech_no_tech[0,:] = mean_median_yield_2020_2099 - (np.sqrt(MC_time_series)+np.sqrt(MS_time_series)+np.sqrt(ST_time_series)+np.sqrt(SR_time_series)+np.sqrt(Intvar_time_series))
y_upper_tech_no_tech[1,:] = mean_median_yield_2020_2099 + (np.sqrt(ST_time_series)+np.sqrt(SR_time_series)+np.sqrt(MC_time_series)+np.sqrt(MS_time_series))
y_lower_tech_no_tech[1,:] = mean_median_yield_2020_2099 - (np.sqrt(ST_time_series)+np.sqrt(SR_time_series)+np.sqrt(MC_time_series)+np.sqrt(MS_time_series))
y_upper_tech_no_tech[2,:] = mean_median_yield_2020_2099 + (np.sqrt(ST_time_series)+np.sqrt(SR_time_series)+np.sqrt(MC_time_series))
y_lower_tech_no_tech[2,:] = mean_median_yield_2020_2099 - (np.sqrt(ST_time_series)+np.sqrt(SR_time_series)+np.sqrt(MC_time_series))
y_upper_tech_no_tech[3,:] = mean_median_yield_2020_2099 + (np.sqrt(ST_time_series)+np.sqrt(SR_time_series))
y_lower_tech_no_tech[3,:] = mean_median_yield_2020_2099 - (np.sqrt(ST_time_series)+np.sqrt(SR_time_series))
y_upper_tech_no_tech[4,:] = mean_median_yield_2020_2099 + np.sqrt(ST_time_series)
y_lower_tech_no_tech[4,:] = mean_median_yield_2020_2099 - np.sqrt(ST_time_series)
#########With tech imrovement
MC_ssp245 = np.zeros(80)
MC_ssp585 = np.zeros(80)
for year in range(0,80):
    MC_ssp245[year] = np.var(np.median(yield_all_sum_ssp245_poly, axis = 2)[year,:])
    MC_ssp585[year] = np.var(np.median(yield_all_sum_ssp585_poly, axis = 2)[year,:])
        
MC_time_series_tech = (MC_ssp245+MC_ssp585)/2

## calculate stat model uncertainty MS
MS_ssp245 = np.zeros(80)
MS_ssp585 = np.zeros(80)
for year in range(0,80):
        MS_ssp245[year] = np.var(np.mean(yield_all_sum_ssp245_poly, axis = 1)[year,:])
        MS_ssp585[year] = np.var(np.mean(yield_all_sum_ssp585_poly, axis = 1)[year,:])  
MS_time_series_tech = (MS_ssp245 + MS_ssp585)/2




## calculate rcp scenario uncertainty SR
yield_all_sum_rcp_scenario_tech = np.zeros((80,8,1000,2))    
yield_all_sum_rcp_scenario_tech[:,:,:,0] = yield_all_sum_ssp245_poly
yield_all_sum_rcp_scenario_tech[:,:,:,1] = yield_all_sum_ssp585_poly


SR_tech = np.zeros(80)
for year in range(0,80):
    SR_tech[year] = np.var(np.median(np.mean(yield_all_sum_rcp_scenario_tech, axis = 1), axis = 1)[year,:])
SR_time_series_tech = SR_tech


## calculate internal 
Intvar_ssp245 = np.zeros(80)
Intvar_ssp585 = np.zeros(80)
residual_ssp245 = np.zeros((80,8,1000))
residual_ssp585 = np.zeros((80,8,1000))
for trial in range(0,1000):
    for climate in range(0,8):
        residual_ssp245[:,climate,trial] = yield_all_sum_ssp245_poly[:,climate,trial]-np.array(pd.DataFrame(yield_all_sum_ssp245_2020_2099.reshape(80,8,1000)[:,climate,trial]).rolling(10,min_periods=5, center = True).mean()).reshape(80)
        residual_ssp585[:,climate,trial] =  yield_all_sum_ssp585_poly[:,climate,trial]-np.array(pd.DataFrame(yield_all_sum_ssp585_2020_2099.reshape(80,8,1000)[:,climate,trial]).rolling(10,min_periods=5, center = True).mean()).reshape(80)


for year in range(0,80):
    Intvar_ssp245[year] = np.var(residual_ssp245[year,:,:])
    Intvar_ssp585[year] = np.var(residual_ssp585[year,:,:])
Intvar_time_series_tech = ((Intvar_ssp245+Intvar_ssp585)/2)

mean_median_yield_2020_2099_tech = np.array((mean_median_yield_2020_2099_ssp245_10_yr_running_mean  + mean_median_yield_2020_2099_ssp585_10_yr_running_mean)/2).reshape(80)

y_upper_tech = np.zeros((4,80))
y_lower_tech= np.zeros((4,80))
y_upper_tech[0,:] = mean_median_yield_2020_2099_tech + (np.sqrt(Intvar_time_series_tech)+np.sqrt(MS_time_series_tech)+np.sqrt(SR_time_series_tech)+np.sqrt(MC_time_series_tech))
y_lower_tech[0,:] = mean_median_yield_2020_2099_tech - (np.sqrt(Intvar_time_series_tech)+np.sqrt(MS_time_series_tech)+np.sqrt(SR_time_series_tech)+np.sqrt(MC_time_series_tech))
y_upper_tech[1,:] = mean_median_yield_2020_2099_tech + (np.sqrt(SR_time_series_tech)+np.sqrt(MC_time_series_tech)+np.sqrt(MS_time_series_tech))
y_lower_tech[1,:] = mean_median_yield_2020_2099_tech - (np.sqrt(SR_time_series_tech)+np.sqrt(MC_time_series_tech)+np.sqrt(MS_time_series_tech))
y_upper_tech[2,:] = mean_median_yield_2020_2099_tech + (np.sqrt(SR_time_series_tech)+np.sqrt(MC_time_series_tech))
y_lower_tech[2,:] = mean_median_yield_2020_2099_tech - (np.sqrt(SR_time_series_tech)+np.sqrt(MC_time_series_tech))
y_upper_tech[3,:] = mean_median_yield_2020_2099_tech + np.sqrt(SR_time_series_tech)
y_lower_tech[3,:] = mean_median_yield_2020_2099_tech - np.sqrt(SR_time_series_tech)



###### Without tech imrovement
##calcualte climate uncertainty
MC_ssp245_s = np.zeros(80)
MC_ssp585_s = np.zeros(80) 
for year in range(0,80):
    MC_ssp245_s[year] = np.var(np.median(yield_all_sum_ssp245_s_poly, axis = 2)[year,:])
    MC_ssp585_s[year] = np.var(np.median(yield_all_sum_ssp585_s_poly, axis = 2)[year,:])
        
MC_time_series_no_tech = (MC_ssp245_s+MC_ssp585_s)/2

## calculate stat model uncertainty MS
MS_ssp245_s = np.zeros(80)
MS_ssp585_s = np.zeros(80) 
for year in range(0,80):
        MS_ssp245_s[year] = np.var(np.mean(yield_all_sum_ssp245_s_poly, axis = 1)[year,:])
        MS_ssp585_s[year] = np.var(np.mean(yield_all_sum_ssp585_s_poly, axis = 1)[year,:])
MS_time_series_no_tech = (MS_ssp245_s + MS_ssp585_s)/2
MC_ssp245_s = np.zeros(80)
MC_ssp585_s = np.zeros(80) 
for year in range(0,80):
    MC_ssp245_s[year] = np.var(np.median(yield_all_sum_ssp245_s_poly, axis = 2)[year,:])
    MC_ssp585_s[year] = np.var(np.median(yield_all_sum_ssp585_s_poly, axis = 2)[year,:])
        
MC_time_series_no_tech = (MC_ssp245_s+MC_ssp585_s)/2

## calculate stat model uncertainty MS
MS_ssp245_s = np.zeros(80)
MS_ssp585_s = np.zeros(80) 
for year in range(0,80):
        MS_ssp245_s[year] = np.var(np.mean(yield_all_sum_ssp245_s_poly, axis = 1)[year,:])
        MS_ssp585_s[year] = np.var(np.mean(yield_all_sum_ssp585_s_poly, axis = 1)[year,:])
MS_time_series_no_tech = (MS_ssp245_s + MS_ssp585_s)/2



## calculate rcp scenario uncertainty SR
yield_all_sum_rcp_scenario_no_tech = np.zeros((80,8,1000,2))    
yield_all_sum_rcp_scenario_no_tech[:,:,:,0] = yield_all_sum_ssp245_s_poly
yield_all_sum_rcp_scenario_no_tech[:,:,:,1] = yield_all_sum_ssp585_s_poly

SR_no_tech = np.zeros(80)
for year in range(0,80):
    SR_no_tech[year] = np.var(np.median(np.mean(yield_all_sum_rcp_scenario_no_tech, axis = 1), axis = 1)[year,:])
SR_time_series_no_tech = SR_no_tech


## calculate internal 
Intvar_ssp245_s = np.zeros(80)
Intvar_ssp585_s = np.zeros(80)
residual_ssp245_s = np.zeros((80,8,1000))
residual_ssp585_s = np.zeros((80,8,1000))


for trial in range(0,1000):
    for climate in range(0,8):
        residual_ssp245_s[:,climate,trial] =  yield_all_sum_ssp245_s_poly[:,climate,trial]-np.array(pd.DataFrame(yield_all_sum_ssp245_s_2020_2099.reshape(80,8,1000)[:,climate,trial]).rolling(10,min_periods=5, center = True).mean()).reshape(80)
        residual_ssp585_s[:,climate,trial] =  yield_all_sum_ssp585_s_poly[:,climate,trial]-np.array(pd.DataFrame(yield_all_sum_ssp585_s_2020_2099.reshape(80,8,1000)[:,climate,trial]).rolling(10,min_periods=5, center = True).mean()).reshape(80)


for year in range(0,80):
    Intvar_ssp245_s[year] = np.var(residual_ssp245_s[year,:,:])
    Intvar_ssp585_s[year] = np.var(residual_ssp585_s[year,:,:])
Intvar_time_series_no_tech = ((Intvar_ssp245_s+Intvar_ssp585_s)/2)

mean_median_yield_2020_2099_no_tech = np.array((mean_median_yield_2020_2099_ssp245_s_10_yr_running_mean  + mean_median_yield_2020_2099_ssp585_s_10_yr_running_mean)/2).reshape(80)
y_upper_no_tech = np.zeros((4,80))
y_lower_no_tech= np.zeros((4,80))
y_upper_no_tech[0,:] = mean_median_yield_2020_2099_no_tech + (np.sqrt(Intvar_time_series_no_tech)+np.sqrt(MS_time_series_no_tech)+np.sqrt(SR_time_series_no_tech)+np.sqrt(MC_time_series_no_tech))
y_lower_no_tech[0,:] = mean_median_yield_2020_2099_no_tech - (np.sqrt(Intvar_time_series_no_tech)+np.sqrt(MS_time_series_no_tech)+np.sqrt(SR_time_series_no_tech)+np.sqrt(MC_time_series_no_tech))
y_upper_no_tech[1,:] = mean_median_yield_2020_2099_no_tech + (np.sqrt(SR_time_series_no_tech)+np.sqrt(MC_time_series_no_tech)+np.sqrt(MS_time_series_no_tech))
y_lower_no_tech[1,:] = mean_median_yield_2020_2099_no_tech - (np.sqrt(SR_time_series_no_tech)+np.sqrt(MC_time_series_no_tech)+np.sqrt(MS_time_series_no_tech))
y_upper_no_tech[2,:] = mean_median_yield_2020_2099_no_tech + (np.sqrt(SR_time_series_no_tech)+np.sqrt(MC_time_series_no_tech))
y_lower_no_tech[2,:] = mean_median_yield_2020_2099_no_tech - (np.sqrt(SR_time_series_no_tech)+np.sqrt(MC_time_series_no_tech))
y_upper_no_tech[3,:] = mean_median_yield_2020_2099_no_tech + np.sqrt(SR_time_series_no_tech)
y_lower_no_tech[3,:] = mean_median_yield_2020_2099_no_tech - np.sqrt(SR_time_series_no_tech)

fig = plt.figure(figsize = (40,10))
plt.subplot(1,3,2)
Var_total_10yr_mean = np.zeros((80,5))
Var_total_10yr_mean[:,0] = MC_time_series
Var_total_10yr_mean[:,1] = MS_time_series
Var_total_10yr_mean[:,2] = ST_time_series
Var_total_10yr_mean[:,3] = SR_time_series
Var_total_10yr_mean[:,4] = Intvar_time_series
for year in range(0,80):
    Var_total_10yr_mean[year,:] = Var_total_10yr_mean[year,:] * 100 / np.sum(Var_total_10yr_mean[year,:])
plt.stackplot(np.arange(2020,2100), Var_total_10yr_mean[:,0],Var_total_10yr_mean[:,1],Var_total_10yr_mean[:,2]
              ,Var_total_10yr_mean[:,3],Var_total_10yr_mean[:,4],labels = ['Climate Model', 'Stats Model', 'Innovation Scenario','SSP Scenario','Internal'], colors = ['#354AA1', '#85B1D4','lightgreen', '#007F3C', '#FF6E04'])
plt.ylim(0,100)
plt.xlim(2020,2100)
plt.xticks(fontsize =35, rotation = 35)
plt.yticks(np.linspace(20,100,5).astype(int),np.linspace(20,100,5).astype(int),fontsize = 35)
plt.title('Fractional contribution to total uncertainty' , fontsize = 30, y = 1.085)
plt.xlabel('Year', fontsize = 35, labelpad = 20)
plt.ylabel('%', fontsize=35)
plt.text(2005, 100, 'b', fontsize = 35, fontweight='bold')

plt.subplot(1,3,1)
plt.fill_between(np.arange(2020,2100), y1 = y_upper_tech_no_tech[0,:], y2 = y_lower_tech_no_tech[0,:],color = '#FF6E04',  label = 'Internal Variability')
plt.fill_between(np.arange(2020,2100), y1 = y_upper_tech_no_tech[1,:], y2 = y_lower_tech_no_tech[1,:],color = '#85B1D4',  label = 'Statistical Model')
plt.fill_between(np.arange(2020,2100), y1 = y_upper_tech_no_tech[2,:], y2 = y_lower_tech_no_tech[2,:],color = '#354AA1', label = 'Climate Model')
plt.fill_between(np.arange(2020,2100), y1 = y_upper_tech_no_tech[3,:], y2 = y_lower_tech_no_tech[3,:],color = '#007F3C',  label = 'SSP Scenario')
plt.fill_between(np.arange(2020,2100), y1 = y_upper_tech_no_tech[4,:], y2 = y_lower_tech_no_tech[4,:],color = 'lightgreen',  label = 'Innovation Scenario')
plt.xticks(fontsize =35, rotation = 35)
plt.yticks(np.linspace(0.5,2.5,5), np.linspace(0.5,2.5,5),fontsize =35)
plt.ylabel('Ton/acre', fontsize = 35)
plt.xlim(2020,2100)
plt.title('Source of yield uncertainty', fontsize = 30, y = 1.085)
plt.legend(loc='upper right', bbox_to_anchor=(3.45, -0.3), ncol = 5, fontsize = 30, edgecolor = 'white')
plt.ylim(0,2.6)
plt.text(2005, 2.6, 'a', fontsize = 35, fontweight='bold')

plt.subplot(1,3,3)
Var_total_10yr_mean = np.zeros((80,4))
Var_total_10yr_mean[:,0] = MC_time_series_no_tech
Var_total_10yr_mean[:,1] = MS_time_series_no_tech
Var_total_10yr_mean[:,2] = SR_time_series_no_tech
Var_total_10yr_mean[:,3] = Intvar_time_series_no_tech
for year in range(0,80):
    Var_total_10yr_mean[year,:] = Var_total_10yr_mean[year,:] * 100 / np.sum(Var_total_10yr_mean[year,:])
plt.stackplot(np.arange(2020,2100), Var_total_10yr_mean[:,0],Var_total_10yr_mean[:,1],Var_total_10yr_mean[:,2]
              ,Var_total_10yr_mean[:,3], labels = ['Climate Model', 'Stats Model', 'SSP Scenario','Internal'], colors = ['#354AA1', '#85B1D4', '#007F3C', '#FF6E04'])
plt.ylim(0,100)
plt.xlim(2020,2100)
plt.xticks(fontsize =35, rotation = 35)
plt.yticks(np.linspace(20,100,5).astype(int),np.linspace(20,100,5).astype(int),fontsize = 35)
plt.title('Fractional contribution to total uncertainty \n with climate change impacts only' , fontsize = 30, y = 1.03)
plt.ylabel('%', fontsize=35)
plt.text(2005, 100, 'c', fontsize = 35, fontweight='bold')
fig.subplots_adjust(wspace=0.35)
plt.savefig(save_path+'uncertainty.pdf', bbox_inches='tight', dpi =300)




##Figure 6: State-level waterfall ACI analysis
fig, axs = plt.subplots(2,2,figsize=(40,30))
formatting = "{:,.1f}"
from matplotlib.ticker import FuncFormatter
index = np.array(ACI_list)[:]
data = median_ssp245_2041_2060
aci_delete_index = np.where(data == 0)
index = np.delete(index, aci_delete_index)
data = np.delete(data, aci_delete_index)
index=np.array(index)
data=np.array(data)
changes = {'amount' : data}
def money(x, pos):
    return formatting.format(x)
formatter = FuncFormatter(money)    
trans = pd.DataFrame(data=changes,index=index)
blank = trans.amount.cumsum().shift(1).fillna(0)
trans['positive'] = trans['amount'] > 0
total = trans.sum().amount
#trans.loc['net']= total
#blank.loc['net'] = total
step = blank.reset_index(drop=True).repeat(3).shift(-1)
step[1::3] = np.nan
#blank.loc['net'] = 0
trans.loc[trans['positive'] > 1, 'positive'] = 99
trans.loc[trans['positive'] < 0, 'positive'] = 99
trans.loc[(trans['positive'] > 0) & (trans['positive'] < 1), 'positive'] = 99
trans['color'] = trans['positive']
trans.loc[trans['positive'] == 1, 'color'] = '#29EA38' #green_color
trans.loc[trans['positive'] == 0, 'color'] = '#FB3C62' #red_color
trans.loc[trans['positive'] == 99, 'color'] = '#24CAFF' #blue_color
my_colors = list(trans.color)
#my_plot = plt.bar(np.arange(0,len(trans.index))-0.5, blank, width=0.4, color='black')
plt.subplot(2,2,1)
plt.bar(np.arange(0,len(trans.index)), trans.amount, width=0.6, edgecolor = 'black',linewidth = 2,
         bottom=blank, color=my_colors)
plt.plot(np.array(step.index), step.values, 'k', linewidth = 2)
plt.text( -2 , 10, 'a',fontsize = 40,fontweight='bold')
plt.yticks(fontsize = 35)
plt.ylim(-120,10)
y_height = trans.amount.cumsum().shift(1).fillna(0)
temp = list(trans.amount) 
for i in range(len(temp)):
    if (i > 0) & (i < (len(temp) - 1)):
        temp[i] = temp[i] + temp[i-1]
trans['temp'] = temp
plot_max = trans['temp'].max()
plot_min = trans['temp'].min()
if all(i >= 0 for i in temp):
    plot_min = 0
if all(i < 0 for i in temp):
    plot_max = 0
if abs(plot_max) >= abs(plot_min):
    maxmax = abs(plot_max)   
else:
    maxmax = abs(plot_min)
pos_offset = maxmax / 40
plot_offset = maxmax / 15 
loop = 0
for index, row in trans.iterrows():
    if row['amount'] == total:
        y = y_height[loop]
    else:
        y = y_height[loop] + row['amount']
    if row['amount'] > 0:
        y += (pos_offset*2)
        plt.annotate(formatting.format(row['amount']),(loop,y-2),ha="center", color = 'g', fontsize=25)
    else:
        y -= (pos_offset*4)
        plt.annotate(formatting.format(row['amount']),(loop,y+3),ha="center", color = 'r', fontsize=25)
    loop+=1
#plt.xticks(np.arange(0,len(trans)), trans.index, rotation = 90, fontsize = 30)
plt.axhline(0, color='black', linewidth = 2, linestyle="dashed")
for i in (0,2,4,6,8,10,12):
    rect=mpatches.Rectangle([-0.5+i,-120], 1, 130, ec='white', fc='grey', alpha=0.2, clip_on=False)
    axs[0,0].add_patch(rect)
plt.tick_params(axis = 'x' , which = 'both', bottom = False, top = False, labelbottom = False)
plt.text(-3,-65, s = 'SSP245', fontsize = 40, rotation = 'vertical')
plt.text(4.8,20, s = '2040-2059', fontsize = 40)

formatting = "{:,.1f}"
from matplotlib.ticker import FuncFormatter
index = np.array(ACI_list)[:]
data = median_ssp245_2080_2099
aci_delete_index = np.where(data == 0)
index = np.delete(index, aci_delete_index)
data = np.delete(data, aci_delete_index)
index=np.array(index)
data=np.array(data)
changes = {'amount' : data}
def money(x, pos):
    return formatting.format(x)
formatter = FuncFormatter(money)    
trans = pd.DataFrame(data=changes,index=index)
blank = trans.amount.cumsum().shift(1).fillna(0)
trans['positive'] = trans['amount'] > 0
total = trans.sum().amount
#trans.loc['net']= total
#blank.loc['net'] = total
step = blank.reset_index(drop=True).repeat(3).shift(-1)
step[1::3] = np.nan
#blank.loc['net'] = 0
trans.loc[trans['positive'] > 1, 'positive'] = 99
trans.loc[trans['positive'] < 0, 'positive'] = 99
trans.loc[(trans['positive'] > 0) & (trans['positive'] < 1), 'positive'] = 99
trans['color'] = trans['positive']
trans.loc[trans['positive'] == 1, 'color'] = '#29EA38' #green_color
trans.loc[trans['positive'] == 0, 'color'] = '#FB3C62' #red_color
trans.loc[trans['positive'] == 99, 'color'] = '#24CAFF' #blue_color
my_colors = list(trans.color)
#my_plot = plt.bar(np.arange(0,len(trans.index))-0.5, blank, width=0.4, color='black')
plt.subplot(2,2,2)
plt.bar(np.arange(0,len(trans.index)), trans.amount, width=0.6, edgecolor = 'black',linewidth = 2,
         bottom=blank, color=my_colors)
plt.plot(np.array(step.index), step.values, 'k', linewidth = 2)
plt.text( -2 , 10, 'b',fontsize = 40,fontweight='bold')
plt.yticks(fontsize = 35)
plt.ylim(-120,10)
y_height = trans.amount.cumsum().shift(1).fillna(0)
temp = list(trans.amount) 
for i in range(len(temp)):
    if (i > 0) & (i < (len(temp) - 1)):
        temp[i] = temp[i] + temp[i-1]
trans['temp'] = temp
plot_max = trans['temp'].max()
plot_min = trans['temp'].min()
if all(i >= 0 for i in temp):
    plot_min = 0
if all(i < 0 for i in temp):
    plot_max = 0
if abs(plot_max) >= abs(plot_min):
    maxmax = abs(plot_max)   
else:
    maxmax = abs(plot_min)
pos_offset = maxmax / 40
plot_offset = maxmax / 15 
loop = 0
for index, row in trans.iterrows():
    if row['amount'] == total:
        y = y_height[loop]
    else:
        y = y_height[loop] + row['amount']
    if row['amount'] > 0:
        y += (pos_offset*2)
        plt.annotate(formatting.format(row['amount']),(loop,y-2),ha="center", color = 'g', fontsize=25)
    else:
        y -= (pos_offset*4)
        plt.annotate(formatting.format(row['amount']),(loop,y+3),ha="center", color = 'r', fontsize=25)
    loop+=1
#plt.xticks(np.arange(0,len(trans)), trans.index, rotation = 90, fontsize = 30)
plt.axhline(0, color='black', linewidth = 2, linestyle="dashed")
for i in (0,2,4,6,8,10,12):
    rect=mpatches.Rectangle([-0.5+i,-120], 1, 130, ec='white', fc='grey', alpha=0.2, clip_on=False)
    axs[0,1].add_patch(rect)
plt.tick_params(axis = 'x' , which = 'both', bottom = False, top = False, labelbottom = False)
plt.text(4.8,20, s = '2080-2099', fontsize = 40)

formatting = "{:,.1f}"
from matplotlib.ticker import FuncFormatter
index = np.array(ACI_list)[:]
data = median_ssp585_2041_2060
aci_delete_index = np.where(data == 0)
index = np.delete(index, aci_delete_index)
data = np.delete(data, aci_delete_index)
index=np.array(index)
data=np.array(data)
changes = {'amount' : data}
def money(x, pos):
    return formatting.format(x)
formatter = FuncFormatter(money)    
trans = pd.DataFrame(data=changes,index=index)
blank = trans.amount.cumsum().shift(1).fillna(0)
trans['positive'] = trans['amount'] > 0
total = trans.sum().amount
#trans.loc['net']= total
#blank.loc['net'] = total
step = blank.reset_index(drop=True).repeat(3).shift(-1)
step[1::3] = np.nan
#blank.loc['net'] = 0
trans.loc[trans['positive'] > 1, 'positive'] = 99
trans.loc[trans['positive'] < 0, 'positive'] = 99
trans.loc[(trans['positive'] > 0) & (trans['positive'] < 1), 'positive'] = 99
trans['color'] = trans['positive']
trans.loc[trans['positive'] == 1, 'color'] = '#29EA38' #green_color
trans.loc[trans['positive'] == 0, 'color'] = '#FB3C62' #red_color
trans.loc[trans['positive'] == 99, 'color'] = '#24CAFF' #blue_color
my_colors = list(trans.color)
#my_plot = plt.bar(np.arange(0,len(trans.index))-0.5, blank, width=0.4, color='black')
plt.subplot(2,2,3)
plt.bar(np.arange(0,len(trans.index)), trans.amount, width=0.6, edgecolor = 'black',linewidth = 2,
         bottom=blank, color=my_colors)
plt.plot(np.array(step.index), step.values, 'k', linewidth = 2)
plt.text( -2 , 10, 'c',fontsize = 40,fontweight='bold')
plt.yticks(fontsize = 35)
plt.ylim(-120,10)
y_height = trans.amount.cumsum().shift(1).fillna(0)
temp = list(trans.amount) 
for i in range(len(temp)):
    if (i > 0) & (i < (len(temp) - 1)):
        temp[i] = temp[i] + temp[i-1]
trans['temp'] = temp
plot_max = trans['temp'].max()
plot_min = trans['temp'].min()
if all(i >= 0 for i in temp):
    plot_min = 0
if all(i < 0 for i in temp):
    plot_max = 0
if abs(plot_max) >= abs(plot_min):
    maxmax = abs(plot_max)   
else:
    maxmax = abs(plot_min)
pos_offset = maxmax / 40
plot_offset = maxmax / 15 
loop = 0
for index, row in trans.iterrows():
    if row['amount'] == total:
        y = y_height[loop]
    else:
        y = y_height[loop] + row['amount']
    if row['amount'] > 0:
        y += (pos_offset*2)
        plt.annotate(formatting.format(row['amount']),(loop,y-2),ha="center", color = 'g', fontsize=25)
    else:
        y -= (pos_offset*4)
        plt.annotate(formatting.format(row['amount']),(loop,y+3),ha="center", color = 'r', fontsize=25)
    loop+=1
#plt.xticks(np.arange(0,len(trans)), trans.index, rotation = 90, fontsize = 30)
plt.axhline(0, color='black', linewidth = 2, linestyle="dashed")
for i in (0,2,4,6,8,10,12):
    rect=mpatches.Rectangle([-0.5+i,-120], 1, 130, ec='white', fc='grey', alpha=0.2, clip_on=False)
    axs[1,0].add_patch(rect)
plt.xticks(np.arange(0,len(trans)), trans.index, rotation=90, fontsize = 35)
plt.text(-3,-65, s = 'SSP585', fontsize = 40, rotation = 'vertical')
annotate_y = -0.5
plt.annotate('', xy = (0.036, annotate_y), xycoords = 'axes fraction', xytext = (0.225,annotate_y), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Dormancy', xy = (0.05, annotate_y-0.1), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.245, annotate_y), xycoords = 'axes fraction', xytext = (0.69,annotate_y), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Bloom', xy = (0.41,annotate_y-0.1), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.71, annotate_y), xycoords = 'axes fraction', xytext = (0.89,annotate_y), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Growth', xy = (0.735, annotate_y-0.1), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.905, annotate_y), xycoords = 'axes fraction', xytext = (0.975,annotate_y), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Harvest', xy = (0.87, annotate_y-0.1), xycoords = 'axes fraction', fontsize = 35)

formatting = "{:,.1f}"
from matplotlib.ticker import FuncFormatter
index = np.array(ACI_list)[:]
data = median_ssp585_2080_2099
aci_delete_index = np.where(data == 0)
index = np.delete(index, aci_delete_index)
data = np.delete(data, aci_delete_index)
index=np.array(index)
data=np.array(data)
changes = {'amount' : data}
def money(x, pos):
    return formatting.format(x)
formatter = FuncFormatter(money)    
trans = pd.DataFrame(data=changes,index=index)
blank = trans.amount.cumsum().shift(1).fillna(0)
trans['positive'] = trans['amount'] > 0
total = trans.sum().amount
#trans.loc['net']= total
#blank.loc['net'] = total
step = blank.reset_index(drop=True).repeat(3).shift(-1)
step[1::3] = np.nan
#blank.loc['net'] = 0
trans.loc[trans['positive'] > 1, 'positive'] = 99
trans.loc[trans['positive'] < 0, 'positive'] = 99
trans.loc[(trans['positive'] > 0) & (trans['positive'] < 1), 'positive'] = 99
trans['color'] = trans['positive']
trans.loc[trans['positive'] == 1, 'color'] = '#29EA38' #green_color
trans.loc[trans['positive'] == 0, 'color'] = '#FB3C62' #red_color
trans.loc[trans['positive'] == 99, 'color'] = '#24CAFF' #blue_color
my_colors = list(trans.color)
#my_plot = plt.bar(np.arange(0,len(trans.index))-0.5, blank, width=0.4, color='black')
plt.subplot(2,2,4)
plt.bar(np.arange(0,len(trans.index)), trans.amount, width=0.6, edgecolor = 'black',linewidth = 2,
         bottom=blank, color=my_colors)
plt.plot(np.array(step.index), step.values, 'k', linewidth = 2)
plt.text( -2 , 10, 'd',fontsize = 40,fontweight='bold')
plt.yticks(fontsize = 35)
plt.ylim(-120,10)
y_height = trans.amount.cumsum().shift(1).fillna(0)
temp = list(trans.amount) 
for i in range(len(temp)):
    if (i > 0) & (i < (len(temp) - 1)):
        temp[i] = temp[i] + temp[i-1]
trans['temp'] = temp
plot_max = trans['temp'].max()
plot_min = trans['temp'].min()
if all(i >= 0 for i in temp):
    plot_min = 0
if all(i < 0 for i in temp):
    plot_max = 0
if abs(plot_max) >= abs(plot_min):
    maxmax = abs(plot_max)   
else:
    maxmax = abs(plot_min)
pos_offset = maxmax / 40
plot_offset = maxmax / 15 
loop = 0
for index, row in trans.iterrows():
    if row['amount'] == total:
        y = y_height[loop]
    else:
        y = y_height[loop] + row['amount']
    if row['amount'] > 0:
        y += (pos_offset*2)
        plt.annotate(formatting.format(row['amount']),(loop,y-2),ha="center", color = 'g', fontsize=25)
    else:
        y -= (pos_offset*4)
        plt.annotate(formatting.format(row['amount']),(loop,y+3),ha="center", color = 'r', fontsize=25)
    loop+=1
#plt.xticks(np.arange(0,len(trans)), trans.index, rotation = 90, fontsize = 30)
plt.axhline(0, color='black', linewidth = 2, linestyle="dashed")
for i in (0,2,4,6,8,10,12):
    rect=mpatches.Rectangle([-0.5+i,-120], 1, 130, ec='white', fc='grey', alpha=0.2, clip_on=False)
    axs[1,1].add_patch(rect)
plt.xticks(np.arange(0,len(trans)), trans.index, rotation=90, fontsize = 35)
annotate_y = -0.5
plt.annotate('', xy = (0.036, annotate_y), xycoords = 'axes fraction', xytext = (0.225,annotate_y), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Dormancy', xy = (0.05, annotate_y-0.1), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.245, annotate_y), xycoords = 'axes fraction', xytext = (0.69,annotate_y), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Bloom', xy = (0.41,annotate_y-0.1), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.71, annotate_y), xycoords = 'axes fraction', xytext = (0.89,annotate_y), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Growth', xy = (0.735, annotate_y-0.1), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.905, annotate_y), xycoords = 'axes fraction', xytext = (0.975,annotate_y), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Harvest', xy = (0.87, annotate_y-0.1), xycoords = 'axes fraction', fontsize = 35)
plt.savefig(save_path+'waterfall_state_all.pdf', dpi = 300,bbox_inches='tight')



##Figure 7: County-level waterfall ACI analysis for ssp585 mid-century

fig, axs = plt.subplots(17,1,figsize=(27,60), gridspec_kw={'height_ratios': [8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]})
formatting = "{:,.1f}"
from matplotlib.ticker import FuncFormatter
index = np.array(ACI_list)[:]
data = median_ssp585_2041_2060
aci_delete_index = np.where(data == 0)
index = np.delete(index, aci_delete_index)
data = np.delete(data, aci_delete_index)
index=np.array(index)
data=np.array(data)
changes = {'amount' : data}
def money(x, pos):
    return formatting.format(x)
formatter = FuncFormatter(money)    
trans = pd.DataFrame(data=changes,index=index)
blank = trans.amount.cumsum().shift(1).fillna(0)
trans['positive'] = trans['amount'] > 0
total = trans.sum().amount
#trans.loc['net']= total
#blank.loc['net'] = total
step = blank.reset_index(drop=True).repeat(3).shift(-1)
step[1::3] = np.nan
#blank.loc['net'] = 0
trans.loc[trans['positive'] > 1, 'positive'] = 99
trans.loc[trans['positive'] < 0, 'positive'] = 99
trans.loc[(trans['positive'] > 0) & (trans['positive'] < 1), 'positive'] = 99
trans['color'] = trans['positive']
trans.loc[trans['positive'] == 1, 'color'] = '#29EA38' #green_color
trans.loc[trans['positive'] == 0, 'color'] = '#FB3C62' #red_color
trans.loc[trans['positive'] == 99, 'color'] = '#24CAFF' #blue_color
my_colors = list(trans.color)
#my_plot = plt.bar(np.arange(0,len(trans.index))-0.5, blank, width=0.4, color='black')
plt.subplot(17,1,1)
plt.bar(np.arange(0,len(trans.index)), trans.amount, width=0.6, edgecolor = 'black',linewidth = 2,
         bottom=blank, color=my_colors)       
plt.plot(np.array(step.index), step.values, 'k', linewidth = 2)
plt.text(s = 'State', x = -4.2, y = -80, fontsize = 40)
plt.yticks(fontsize = 35)
plt.ylim(-200,50)
plt.yticks([-200,-100,0], [-200,-100,0])
plt.title('SSP585 2040-2059', fontsize = 45, y =1.05)

#plt.xticks(np.arange(0,len(trans)), trans.index, rotation = 90, fontsize = 30)
plt.axhline(0, color='black', linewidth = 0.6, linestyle="dashed")
for i in (0,2,4,6,8,10,12):
    rect=mpatches.Rectangle([-0.5+i,-120], 1, 130, ec='white', fc='grey', alpha=0.2, clip_on=False)
    axs[0].add_patch(rect)
plt.tick_params(axis = 'x' , which = 'both', bottom = False, top = False, labelbottom = False)
county_order_N_S = ['Tehama', 'Butte', 'Glenn', 'Yuba', 'Sutter', 'Colusa', 'Yolo', 'Solano', 'San Joaquin', 'Stanislaus', 'Madera', 'Merced', 'Fresno', 'Tulare', 'Kings', 'Kern']
county_order_N_S_num = [12, 0, 3, 15, 11, 1, 14, 9, 8, 10, 6, 7, 2,13, 5, 4]
for county in range(0,16):     
    formatting = "{:,.1f}"
    from matplotlib.ticker import FuncFormatter
    index = np.array(ACI_list)[:]
    data = aci_contribution_ssp585_county_2050_change_percent_median[county_order_N_S_num[county]]
    index = np.delete(index, aci_delete_index)
    data = np.delete(data, aci_delete_index)
    index=np.array(index)
    data=np.array(data)
    changes = {'amount' : data}
    def money(x, pos):
        return formatting.format(x)
    formatter = FuncFormatter(money)    
    trans = pd.DataFrame(data=changes,index=index)
    blank = trans.amount.cumsum().shift(1).fillna(0)
    trans['positive'] = trans['amount'] > 0
    total = trans.sum().amount
    step = blank.reset_index(drop=True).repeat(3).shift(-1)
    step[1::3] = np.nan
    trans.loc[trans['positive'] > 1, 'positive'] = 99
    trans.loc[trans['positive'] < 0, 'positive'] = 99
    trans.loc[(trans['positive'] > 0) & (trans['positive'] < 1), 'positive'] = 99
    trans['color'] = trans['positive']
    trans.loc[trans['positive'] == 1, 'color'] = '#29EA38' #green_color
    trans.loc[trans['positive'] == 0, 'color'] = '#FB3C62' #red_color
    trans.loc[trans['positive'] == 99, 'color'] = '#24CAFF' #blue_color
    my_colors = list(trans.color)
    plt.subplot(17,1,county+2)
    plt.text(s = str(county_order_N_S[county]), x = -4.2,y = -80, fontsize = 40)
    plt.plot(np.array(step.index), step.values, 'k', linewidth = 2)
    plt.subplot(17,1,county+2)
    plt.bar(range(0,len(trans.index)), trans.amount, width=0.6, edgecolor = 'black',linewidth = 2,
             bottom=blank, color=my_colors)       
    plt.yticks(fontsize = 25)
    y_height = trans.amount.cumsum().shift(1).fillna(0)
    temp = list(trans.amount) 
    for i in range(len(temp)):
        if (i > 0) & (i < (len(temp) - 1)):
            temp[i] = temp[i] + temp[i-1]
    trans['temp'] = temp
    plot_max = trans['temp'].max()
    plot_min = trans['temp'].min()
    if all(i >= 0 for i in temp):
        plot_min = 0
    if all(i < 0 for i in temp):
        plot_max = 0
    if abs(plot_max) >= abs(plot_min):
        maxmax = abs(plot_max)   
    else:
        maxmax = abs(plot_min)
    pos_offset = maxmax / 40
    plot_offset = maxmax / 15 
    loop = 0
    #for index, row in trans.iterrows():
     #   if row['amount'] == total:
      #      y = y_height[loop]
       # else:
        #    y = y_height[loop] + row['amount']
        #if row['amount'] > 0:
         #   y += (pos_offset*2)
          #  plt.annotate(formatting.format(row['amount']),(loop,y),ha="center", color = 'g', fontsize=20)
        #else:
         #   y -= (pos_offset*4)
          #  plt.annotate(formatting.format(row['amount']),(loop,y-25),ha="center", color = 'r', fontsize=20)
        #loop+=1
    plt.ylim(-200,50)
    plt.yticks([-200,-100,0], [-200,-100,0], fontsize = 35)
    plt.axhline(0, color='black', linewidth = 0.6, linestyle="dashed")
    if county == 15:
        plt.xticks(np.arange(0,len(trans)), trans.index, rotation=90, fontsize = 35)
    else:
        plt.tick_params(axis = 'x' , which = 'both', bottom = False, top = False, labelbottom = False)
    for j in (0,2,4,6,8,10,12):
        rect=mpatches.Rectangle((-0.5+j,-200), 1, 250, ec='black', fc='grey', alpha=0.2, clip_on=False)
        axs[county+1].add_patch(rect)
plt.annotate('', xy = (0.050, -2.3), xycoords = 'axes fraction', xytext = (0.22,-2.3), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Dormancy', xy = (0.085, -2.6), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.255, -2.3), xycoords = 'axes fraction', xytext = (0.685,-2.3), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Bloom', xy = (0.44,-2.6), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.72, -2.3), xycoords = 'axes fraction', xytext = (0.885,-2.3), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Growth', xy = (0.77, -2.6), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.91, -2.3), xycoords = 'axes fraction', xytext = (0.97,-2.3), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Harvest', xy = (0.9, -2.6), xycoords = 'axes fraction', fontsize = 35)
plt.savefig(save_path+'waterfall_total_2050_ssp585.pdf', dpi = 300,bbox_inches='tight')


