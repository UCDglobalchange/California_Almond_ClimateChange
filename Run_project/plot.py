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

home_path=
input_path_gridmet = home_path+'/intermediate_data/Gridmet_csv/'
input_path_projection = home_path+'/output_data/projection/'
input_path_contribution = home_path+'/output_data/aci_contribution/'
input_path_model = home_path+'/intermediate_data/lasso_model/'
input_path = home_path+'/input_data/'
save_path = home_path+'/output_data/plots/'
shp_path = home_path+'/input_data/CA_Counties/'

## Load yield simulations
yield_all_future_rcp45 = np.load(input_path_projection+'yield_all_future_rcp45.npy')
yield_all_future_rcp45_s = np.load(input_path_projection+'yield_all_future_rcp45_s.npy')
yield_all_future_rcp45_m = np.load(input_path_projection+'yield_all_future_rcp45_m.npy')
yield_all_future_rcp85 = np.load(input_path_projection+'yield_all_future_rcp85.npy')
yield_all_future_rcp85_s = np.load(input_path_projection+'yield_all_future_rcp85_s.npy')
yield_all_future_rcp85_m = np.load(input_path_projection+'yield_all_future_rcp85_m.npy')
yield_all_hist_rcp45 = np.load(input_path_projection+'yield_all_hist_rcp45.npy')
yield_all_hist_rcp45_s = np.load(input_path_projection+'yield_all_hist_rcp45_s.npy')
yield_all_hist_rcp45_m = np.load(input_path_projection+'yield_all_hist_rcp45_m.npy')
yield_all_hist_rcp85 = np.load(input_path_projection+'yield_all_hist_rcp85.npy')
yield_all_hist_rcp85_s = np.load(input_path_projection+'yield_all_hist_rcp85_s.npy')
yield_all_hist_rcp85_m = np.load(input_path_projection+'yield_all_hist_rcp85_m.npy')

yield_across_state_hist_rcp45 = np.load(input_path_projection+'yield_across_state_hist_rcp45.npy')
yield_across_state_hist_rcp45_s = np.load(input_path_projection+'yield_across_state_hist_rcp45_s.npy')
yield_across_state_hist_rcp85 = np.load(input_path_projection+'yield_across_state_hist_rcp85.npy')
yield_across_state_hist_rcp85_s = np.load(input_path_projection+'yield_across_state_hist_rcp85_s.npy')
yield_across_state_future_rcp45 = np.load(input_path_projection+'yield_across_state_future_rcp45.npy')
yield_across_state_future_rcp45_s = np.load(input_path_projection+'yield_across_state_future_rcp45_s.npy')
yield_across_state_future_rcp85 = np.load(input_path_projection+'yield_across_state_future_rcp85.npy')
yield_across_state_future_rcp85_s = np.load(input_path_projection+'yield_across_state_future_rcp85_s.npy')

yield_average_model_hist_rcp45 = np.load(input_path_projection+'yield_average_model_hist_rcp45.npy')
yield_average_model_hist_rcp45_s = np.load(input_path_projection+'yield_average_model_hist_rcp45_s.npy')
yield_average_model_hist_rcp85 = np.load(input_path_projection+'yield_average_model_hist_rcp85.npy')
yield_average_model_hist_rcp85_s = np.load(input_path_projection+'yield_average_model_hist_rcp85_s.npy')
yield_average_model_future_rcp45 = np.load(input_path_projection+'yield_average_model_future_rcp45.npy')
yield_average_model_future_rcp45_s = np.load(input_path_projection+'yield_average_model_future_rcp45_s.npy')
yield_average_model_future_rcp85 = np.load(input_path_projection+'yield_average_model_future_rcp85.npy')
yield_average_model_future_rcp85_s = np.load(input_path_projection+'yield_average_model_future_rcp85_s.npy')

yield_all_hist_rcp45 = np.load(input_path_projection+'yield_all_hist_rcp45.npy')
yield_all_hist_rcp45_s = np.load(input_path_projection+'yield_all_hist_rcp45_s.npy')
yield_all_hist_rcp85 = np.load(input_path_projection+'yield_all_hist_rcp85.npy')
yield_all_hist_rcp85_s = np.load(input_path_projection+'yield_all_hist_rcp85_s.npy')

yield_all_model_hist_rcp45 = np.load(input_path_projection+'yield_all_model_hist_rcp45.npy')
yield_all_model_hist_rcp45_s = np.load(input_path_projection+'yield_all_model_hist_rcp45_s.npy')
yield_all_model_future_rcp45 = np.load(input_path_projection+'yield_all_model_future_rcp45.npy')
yield_all_model_future_rcp45_s = np.load(input_path_projection+'yield_all_model_future_rcp45_s.npy')

yield_all_model_hist_rcp85 = np.load(input_path_projection+'yield_all_model_hist_rcp85.npy')
yield_all_model_hist_rcp85_s = np.load(input_path_projection+'yield_all_model_hist_rcp85_s.npy')
yield_all_model_future_rcp85 = np.load(input_path_projection+'yield_all_model_future_rcp85.npy')
yield_all_model_future_rcp85_s = np.load(input_path_projection+'yield_all_model_future_rcp85_s.npy')

##load coefficient from lasso 1000 models 
aci_num = 13
for i in range(1,11):
    locals()['coef'+str(i)] = np.zeros((100,aci_num*2+32))
coef_sum = np.zeros((0,aci_num*2+32))
for i in range(1,1001):
    print(i)
    locals()['coef'+str(((i-1)//100)+1)][i%100-1] = genfromtxt(input_path_model+'coef_'+str(i)+'.csv', delimiter = ',')
for i in range(1,11):
    coef_sum = np.row_stack((coef_sum, locals()['coef'+str(i)]))




area = genfromtxt(input_path+'almond_area.csv', delimiter = ',')
production = genfromtxt(input_path+'almond_production.csv', delimiter = ',')
gridmet = genfromtxt(input_path_gridmet+'Gridmet.csv', delimiter = ',')
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

future_tech_trend_county_rcp45_con = np.zeros((120,16,1000))
future_tech_trend_county_rcp45_int = np.zeros((120,16,1000))
for i in range(16):
    for year in range(0,120):
        for trial in range(1000):
            future_tech_trend_county_rcp45_con[year,i,trial] = tech_trend_county_con[i,year,trial] + np.mean(np.split(yield_all_model_hist_rcp45_s,16)[i][-1,:].reshape(18,1000), axis=0)[trial]
            future_tech_trend_county_rcp45_int[year,i,trial] = tech_trend_county_int[i,year,trial] + np.mean(np.split(yield_all_model_hist_rcp45_s,16)[i][-1,:].reshape(18,1000), axis=0)[trial]

##Figure 1: yield time series
df_2080_2099_20yrmean_yield_rcp45 = pd.DataFrame({'scenario' : np.repeat('rcp45', 18000),'mean_yield' : np.mean(yield_all_future_rcp45[-20:],axis=0), 'tech' : np.repeat('yes',18000)})
df_2080_2099_20yrmean_yield_rcp45_s = pd.DataFrame({'scenario' : np.repeat('rcp45', 18000),'mean_yield' : np.mean(yield_all_future_rcp45_s[-20:],axis=0), 'tech' : np.repeat('no',18000)})
df_2080_2099_20yrmean_yield_rcp45_m = pd.DataFrame({'scenario' : np.repeat('rcp45', 18000),'mean_yield' : np.mean(yield_all_future_rcp45_m[-20:],axis=0), 'tech' : np.repeat('m',18000)})
df_2080_2099_20yrmean_yield_rcp85 = pd.DataFrame({'scenario' : np.repeat('rcp85', 18000),'mean_yield' : np.mean(yield_all_future_rcp85[-20:],axis=0), 'tech' : np.repeat('yes',18000)})
df_2080_2099_20yrmean_yield_rcp85_s = pd.DataFrame({'scenario' : np.repeat('rcp85', 18000),'mean_yield' : np.mean(yield_all_future_rcp85_s[-20:],axis=0), 'tech' : np.repeat('no',18000)})
df_2080_2099_20yrmean_yield_rcp85_m = pd.DataFrame({'scenario' : np.repeat('rcp85', 18000),'mean_yield' : np.mean(yield_all_future_rcp85_m[-20:],axis=0), 'tech' : np.repeat('m',18000)})
df_2080_2099_20yrmean_yield_rcp45_total = pd.concat((df_2080_2099_20yrmean_yield_rcp45, df_2080_2099_20yrmean_yield_rcp45_s, df_2080_2099_20yrmean_yield_rcp45_m))
df_2080_2099_20yrmean_yield_rcp85_total = pd.concat((df_2080_2099_20yrmean_yield_rcp85, df_2080_2099_20yrmean_yield_rcp85_s, df_2080_2099_20yrmean_yield_rcp85_m))

yield_change_to_simulate2020_rcp45 = ((np.mean(yield_all_future_rcp45[-20:],axis=0) - yield_all_hist_rcp45[-1])*100/yield_all_hist_rcp45[-1]).reshape(18,1000)
yield_change_to_simulate2020_rcp85 = ((np.mean(yield_all_future_rcp85[-20:],axis=0) - yield_all_hist_rcp85[-1])*100/yield_all_hist_rcp85[-1]).reshape(18,1000)
yield_change_to_simulate2020_rcp45_s = ((np.mean(yield_all_future_rcp45_s[-20:],axis=0) - yield_all_hist_rcp45[-1])*100/yield_all_hist_rcp45[-1]).reshape(18,1000)
yield_change_to_simulate2020_rcp85_s = ((np.mean(yield_all_future_rcp85_s[-20:],axis=0) - yield_all_hist_rcp85[-1])*100/yield_all_hist_rcp85[-1]).reshape(18,1000)
yield_change_to_simulate2020_rcp45_m = ((np.mean(yield_all_future_rcp45_m[-20:],axis=0) - yield_all_hist_rcp45[-1])*100/yield_all_hist_rcp45[-1]).reshape(18,1000)
yield_change_to_simulate2020_rcp85_m = ((np.mean(yield_all_future_rcp85_m[-20:],axis=0) - yield_all_hist_rcp85[-1])*100/yield_all_hist_rcp85[-1]).reshape(18,1000)

yield_change_to_simulate2020_rcp45_ave = np.median(np.mean(yield_change_to_simulate2020_rcp45, axis=0), axis = 0)
yield_change_to_simulate2020_rcp85_ave = np.median(np.mean(yield_change_to_simulate2020_rcp85, axis=0), axis = 0)
yield_change_to_simulate2020_rcp45_s_ave = np.median(np.mean(yield_change_to_simulate2020_rcp45_s, axis=0), axis = 0)
yield_change_to_simulate2020_rcp85_s_ave = np.median(np.mean(yield_change_to_simulate2020_rcp85_s, axis=0), axis = 0)
yield_change_to_simulate2020_rcp45_m_ave = np.median(np.mean(yield_change_to_simulate2020_rcp45_m, axis=0), axis = 0)
yield_change_to_simulate2020_rcp85_m_ave = np.median(np.mean(yield_change_to_simulate2020_rcp85_m, axis=0), axis = 0)

##calculate num of year with % loss 
def get_num_year_with_loss(obs, pred):
    loss_percent_axis = np.linspace(0, 1, 21)
    matrix = np.zeros(len(loss_percent_axis))
    for i in range(len(loss_percent_axis)):
        matrix[i] = np.count_nonzero(pred<=(1-loss_percent_axis[i])*obs)
    return(matrix)

def extract_prob_curve(matrix, prob):
    min_value = np.min(np.abs(matrix-prob),axis=0)
    min_value_index = np.argmin(np.abs(matrix-prob),axis=0)
    if min_value_index[-1] < min_value_index[-2]:
        min_value_index[-1] = min_value_index[-2]
    df = pd.DataFrame({'x' : np.arange(0,21), 'y' : min_value, 'min_index' : min_value_index})
    df = df.loc[df['y'] <=0.1]
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
                        

matrix_percent_loss_num_year_rcp45_2090 = np.zeros((21,18000))
matrix_percent_loss_num_year_rcp85_2090 = np.zeros((21,18000))
matrix_percent_loss_num_year_rcp45_2050 = np.zeros((21,18000))
matrix_percent_loss_num_year_rcp85_2050 = np.zeros((21,18000))
for trial in range(18000):
    matrix_percent_loss_num_year_rcp45_2090[:,trial] = get_num_year_with_loss(yield_all_hist_rcp45_s[-1,trial], yield_all_future_rcp45_s[-20:,trial])
    matrix_percent_loss_num_year_rcp85_2090[:,trial] = get_num_year_with_loss(yield_all_hist_rcp85_s[-1,trial], yield_all_future_rcp85_s[-20:,trial])
    matrix_percent_loss_num_year_rcp45_2050[:,trial] = get_num_year_with_loss(yield_all_hist_rcp45_s[-1,trial], yield_all_future_rcp45_s[19:39,trial])
    matrix_percent_loss_num_year_rcp85_2050[:,trial] = get_num_year_with_loss(yield_all_hist_rcp85_s[-1,trial], yield_all_future_rcp85_s[19:39,trial])
    
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
blue_patch = mpatches.Patch(color = 'royalblue', label = 'Continued historical technological improvement')
red_patch = mpatches.Patch(color = 'lightcoral', label = 'No additional technological improvement past 2020')
grey_patch = mpatches.Patch(color = 'grey', label = 'Historical simulations')
purple_patch = mpatches.Patch(color = 'mediumorchid', label = 'Intermediate technological improvement')
#blank_patch = mpatches.Patch(color = 'white', label = 'Future projections:')
plt.yticks(fontsize = 30)
plt.xlim(1980,2100)
plt.ylim(0,2.5)
plt.ylabel('Yield ton/acre', fontsize = 35)
plt.xticks((1980,1990,2000,2010,2020,2030,2040,2050,2060,2070,2080,2090,2100),['1980','1990', '2000', '2010','2020','2030', '2040', '2050','2060','2070', '2080','2090','2100'],fontsize = 30, rotation = 45)
plt.plot(np.arange(1980,2021), yield_observed_state[0:41], linewidth = 4, color = 'green')
tsplot(np.arange(1980,2021), np.transpose(yield_all_hist_rcp45[0:41]) , color = 'grey')
plt.plot(np.arange(1980,2021), np.median(yield_all_hist_rcp45, axis=1), color = 'black', linewidth = 4)
first_legend = plt.legend(handles=[Line2D([0], [0], color='green', lw=4, label='Observed yield'),grey_patch], fontsize = 30,fancybox=False, shadow=False, ncol = 2, bbox_to_anchor=(1, -1.8), edgecolor = 'white')
plt.gca().add_artist(first_legend)
second_legend = plt.legend(handles=[blue_patch, purple_patch, red_patch],fontsize = 30,fancybox=False, shadow=False, ncol = 1, bbox_to_anchor=(1.11, -1.9), edgecolor = 'white')
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
ax8 = sns.heatmap(prob_percent_loss_year_rcp45_2050, cmap = 'coolwarm', cbar = False, square = True)
ax8.tick_params(axis = 'y', width=2, length = 5)
ax8.tick_params(axis = 'x', width=2, length = 5)
plt.plot(extract_prob_curve(prob_percent_loss_year_rcp45_2050,0.5)[0]+0.5, extract_prob_curve(prob_percent_loss_year_rcp45_2050,0.5)[1]+0.5, color = 'k', linewidth = 5)
plt.plot(extract_prob_curve(prob_percent_loss_year_rcp45_2050,0.9)[0]+0.5, extract_prob_curve(prob_percent_loss_year_rcp45_2050,0.9)[1]+0.5, color = 'k', linewidth = 5, linestyle = 'dashed')
plt.plot(extract_prob_curve(prob_percent_loss_year_rcp45_2050,0.1)[0]+0.5, extract_prob_curve(prob_percent_loss_year_rcp45_2050,0.1)[1]+0.5, color = 'k', linewidth = 5, linestyle = 'dotted')
y_tick_pos = np.array([0,2,4,6,8,10,12,14,16,18])
y_tick_value = (2,4,6,8,10,12,14,16,18,20)
plt.xticks(np.arange(0,21,2)+0.5, np.array(np.linspace(0,100,11)).astype(int), rotation = 45)
plt.xticks(fontsize = 30)
plt.yticks(y_tick_pos+0.5, y_tick_value[::-1], fontsize = 30, rotation = 360)
plt.title('2040-2059', fontsize = 35, y = 1.05)
plt.ylabel('Number of years', fontsize = 35)
ax8.set(xlabel=None)
#ax8.set(xticklabels=[])

ax9 = plt.subplot(spec[7:37,8])
ax9 = sns.heatmap(prob_percent_loss_year_rcp45_2090, cmap = 'coolwarm', square=True, cbar = False)
ax9.tick_params(axis = 'y', width=2, length = 5)
ax9.tick_params(axis = 'x', width=2, length = 5)
plt.plot(extract_prob_curve(prob_percent_loss_year_rcp45_2090,0.5)[0]+0.5, extract_prob_curve(prob_percent_loss_year_rcp45_2090,0.5)[1]+0.5, color = 'k', linewidth = 5)
plt.plot(extract_prob_curve(prob_percent_loss_year_rcp45_2090,0.9)[0]+0.5, extract_prob_curve(prob_percent_loss_year_rcp45_2090,0.9)[1]+0.5, color = 'k', linewidth = 5, linestyle = 'dashed')
plt.plot(extract_prob_curve(prob_percent_loss_year_rcp45_2090,0.1)[0]+0.5, extract_prob_curve(prob_percent_loss_year_rcp45_2090,0.1)[1]+0.5, color = 'k', linewidth = 5, linestyle = 'dotted')
plt.xticks(np.arange(0,21,2)+0.5, np.array(np.linspace(0,100,11)).astype(int), rotation = 45)
plt.xticks(fontsize = 30)
plt.yticks(y_tick_pos+0.5, y_tick_value[::-1], fontsize = 30, rotation = 360)
ax9.set(xlabel=None)
#ax9.set(xticklabels=[])
plt.title('2080-2099', fontsize = 35, y = 1.05)
plt.text(-6.8, -4.25, 'RCP4.5', fontsize = 35)


ax10 = plt.subplot(spec[51:100,6])
ax10 = sns.heatmap(prob_percent_loss_year_rcp85_2050, cmap = 'coolwarm', cbar = False, square = True)
ax10.tick_params(axis = 'y', width=2, length = 5)
ax10.tick_params(axis = 'x', width=2, length = 5)
plt.plot(extract_prob_curve(prob_percent_loss_year_rcp85_2050,0.5)[0]+0.5, extract_prob_curve(prob_percent_loss_year_rcp85_2050,0.5)[1]+0.5, color = 'k', linewidth = 5)
plt.plot(extract_prob_curve(prob_percent_loss_year_rcp85_2050,0.9)[0]+0.5, extract_prob_curve(prob_percent_loss_year_rcp85_2050,0.9)[1]+0.5, color = 'k', linewidth = 5, linestyle = 'dashed')
plt.plot(extract_prob_curve(prob_percent_loss_year_rcp85_2050,0.1)[0]+0.5, extract_prob_curve(prob_percent_loss_year_rcp85_2050,0.1)[1]+0.5, color = 'k', linewidth = 5, linestyle = 'dotted')
y_tick_pos = np.array([0,2,4,6,8,10,12,14,16,18])
y_tick_value = (2,4,6,8,10,12,14,16,18,20)
plt.xticks(np.arange(0,21,2)+0.5, np.array(np.linspace(0,100,11)).astype(int), rotation = 45)
plt.xticks(fontsize = 30)
plt.yticks(y_tick_pos+0.5, y_tick_value[::-1], fontsize = 30, rotation = 360)
plt.ylabel('Number of years', fontsize = 35)
ax10.set_xlabel('Percentage of yield loss from climate change', fontsize = 35)
ax10.xaxis.set_label_coords(1.1, -.2)
plt.title('2040-2059', fontsize = 35, y = 1.05)
#ax10.yaxis.set_label_coords(-0.1, 1.15)

ax11 = plt.subplot(spec[51:100,8])
ax11 = sns.heatmap(prob_percent_loss_year_rcp85_2090, cmap = 'coolwarm', square=True, cbar = False)
ax11.tick_params(axis = 'y', width=2, length = 5)
ax11.tick_params(axis = 'x', width=2, length = 5)
plt.plot(extract_prob_curve(prob_percent_loss_year_rcp85_2090,0.5)[0]+0.5, extract_prob_curve(prob_percent_loss_year_rcp85_2090,0.5)[1]+0.5, color = 'k', linewidth = 5)
plt.plot(extract_prob_curve(prob_percent_loss_year_rcp85_2090,0.9)[0]+0.5, extract_prob_curve(prob_percent_loss_year_rcp85_2090,0.9)[1]+0.5, color = 'k', linewidth = 5, linestyle = 'dashed')
plt.plot(extract_prob_curve(prob_percent_loss_year_rcp85_2090,0.1)[0]+0.5, extract_prob_curve(prob_percent_loss_year_rcp85_2090,0.1)[1]+0.5, color = 'k', linewidth = 5, linestyle = 'dotted')
plt.xticks(np.arange(0,21,2)+0.5, np.array(np.linspace(0,100,11)).astype(int), rotation = 45)
plt.xticks(fontsize = 30)
plt.yticks(y_tick_pos+0.5, y_tick_value[::-1], fontsize = 30, rotation = 360)
cbar = ax10.get_children()[0]
plt.colorbar(cbar, ax = [ax10,ax11], location = 'bottom', pad = 0.26, shrink = 0.8)
cbar = ax10.collections[0].colorbar
cbar.ax.tick_params(labelsize=30)
cbar.set_label('Probability', fontsize = 35, labelpad=0.1)
plt.title('2080-2099', fontsize = 35, y = 1.05)
plt.text(-6.8, -4.25, 'RCP8.5', fontsize = 35)
plt.savefig(save_path+'yield_time_series.pdf', dpi = 300)
plt.show()




## Figure 2: Spatial distribution of yield change
county_list = ['Butte', 'Colusa', 'Fresno', 'Glenn', 'Kern', 'Kings', 'Madera', 'Merced', 'San Joaquin', 'Solano', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Yolo', 'Yuba']
for i in range(0,16):
    locals()[str(county_list[i])+'yield_rcp45'] = np.row_stack((np.split(yield_all_model_hist_rcp45, 16)[i], np.split(yield_all_model_future_rcp45, 16)[i]))
    locals()[str(county_list[i])+'yield_rcp45_s'] = np.row_stack((np.split(yield_all_model_hist_rcp45_s, 16)[i], np.split(yield_all_model_future_rcp45_s, 16)[i]))
    locals()[str(county_list[i])+'yield_rcp85'] = np.row_stack((np.split(yield_all_model_hist_rcp85, 16)[i], np.split(yield_all_model_future_rcp85, 16)[i]))
    locals()[str(county_list[i])+'yield_rcp85_s'] = np.row_stack((np.split(yield_all_model_hist_rcp85_s, 16)[i], np.split(yield_all_model_future_rcp85_s, 16)[i]))


for i in range(0,16):
    locals()[str(county_list[i])+'county_yield_change_2020'] = np.zeros((18000,4))
    locals()[str(county_list[i])+'county_yield_change_2020'][:,0] = locals()[str(county_list[i])+'yield_rcp45'][40,:]
    locals()[str(county_list[i])+'county_yield_change_2020'][:,1] = locals()[str(county_list[i])+'yield_rcp45_s'][40,:]
    locals()[str(county_list[i])+'county_yield_change_2020'][:,2] = locals()[str(county_list[i])+'yield_rcp85'][40,:]
    locals()[str(county_list[i])+'county_yield_change_2020'][:,3] = locals()[str(county_list[i])+'yield_rcp85_s'][40,:]
    locals()[str(county_list[i])+'county_yield_change_2099'] = np.zeros((18000,4))
    locals()[str(county_list[i])+'county_yield_change_2099'][:,0] = ((np.nanmean(locals()[str(county_list[i])+'yield_rcp45'][100:120,:], axis=0))-locals()[str(county_list[i])+'county_yield_change_2020'][:,0])*100/locals()[str(county_list[i])+'county_yield_change_2020'][:,0]
    locals()[str(county_list[i])+'county_yield_change_2099'][:,1] = ((np.nanmean(locals()[str(county_list[i])+'yield_rcp45_s'][100:120,:], axis=0))-locals()[str(county_list[i])+'county_yield_change_2020'][:,1])*100/locals()[str(county_list[i])+'county_yield_change_2020'][:,1]
    locals()[str(county_list[i])+'county_yield_change_2099'][:,2] = ((np.nanmean(locals()[str(county_list[i])+'yield_rcp85'][100:120,:], axis=0))-locals()[str(county_list[i])+'county_yield_change_2020'][:,2])*100/locals()[str(county_list[i])+'county_yield_change_2020'][:,2]
    locals()[str(county_list[i])+'county_yield_change_2099'][:,3] = ((np.nanmean(locals()[str(county_list[i])+'yield_rcp85_s'][100:120,:], axis=0))-locals()[str(county_list[i])+'county_yield_change_2020'][:,3])*100/locals()[str(county_list[i])+'county_yield_change_2020'][:,3]
    locals()[str(county_list[i])+'county_tech_change_2099'] = np.zeros((1000,2))
    locals()[str(county_list[i])+'county_tech_change_2099'][:,0] = 100 * (np.mean(future_tech_trend_county_rcp45_con[100:120,i,:], axis=0) - np.mean(locals()[str(county_list[i])+'county_yield_change_2020'][:,0].reshape(18,1000), axis=0)) / np.mean(locals()[str(county_list[i])+'county_yield_change_2020'][:,0].reshape(18,1000), axis=0)
    locals()[str(county_list[i])+'county_tech_change_2099'][:,1] = 100 * (np.mean(future_tech_trend_county_rcp45_int[100:120,i,:], axis=0) - np.mean(locals()[str(county_list[i])+'county_yield_change_2020'][:,0].reshape(18,1000), axis=0)) / np.mean(locals()[str(county_list[i])+'county_yield_change_2020'][:,0].reshape(18,1000), axis=0)


median_yield_change_2099 = np.zeros((16,4))
median_tech_change_2099 = np.zeros((16,2))

for i in range(0,16):
    median_yield_change_2099[i,:] = np.nanmedian(locals()[str(county_list[i])+'county_yield_change_2099'], axis = 0)
    median_tech_change_2099[i,:] = np.median(locals()[str(county_list[i])+'county_tech_change_2099'], axis=0)
    
yield_change_for_shp_45_2099 = np.zeros((58))
yield_change_for_shp_45_2099[:] = np.nan
yield_change_for_shp_85_2099 = np.zeros((58))
yield_change_for_shp_85_2099[:] = np.nan

tech_change_for_shp_45_2099_con = np.zeros((58))
tech_change_for_shp_45_2099_con[:] = np.nan
tech_change_for_shp_45_2099_int = np.zeros((58))
tech_change_for_shp_45_2099_int[:] = np.nan

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
            yield_change_for_shp_45_2099[i] = median_yield_change_2099[index,1]
            yield_change_for_shp_85_2099[i] = median_yield_change_2099[index,3]
            #yield_change_for_shp_45_2099_value[i] = median_yield_change_2099_value[index,1]
            #yield_change_for_shp_85_2099_value[i] = median_yield_change_2099_value[index,3]
            tech_change_for_shp_45_2099_con[i] = median_tech_change_2099[index, 0]
            tech_change_for_shp_45_2099_int[i] = median_tech_change_2099[index, 1]
            
county_order_N_S = ['Tehama', 'Butte', 'Glenn', 'Yuba', 'Colusa', 'Sutter', 'Yolo', 'Solano', 'San Joaquin', 'Stanislaus', 'Madera', 'Merced', 'Fresno', 'Tulare', 'Kings', 'Kern']

for i in range(0,16):
    N_S_order[np.array(np.where(ca_county_remove_shp['NAME'] == county_order_N_S[i]))] = i+1
ca_county_remove_shp['N_S_order'] = N_S_order.astype(int)

yield_change_for_shp_45_2099_df = pd.DataFrame({'NAME' : ca.NAME, 'rcp45_2099' : yield_change_for_shp_45_2099})
yield_change_for_shp_85_2099_df = pd.DataFrame({'NAME' : ca.NAME, 'rcp85_2099' : yield_change_for_shp_85_2099})
tech_change_for_shp_45_2090_con_df = pd.DataFrame({'NAME' : ca.NAME, 'yield_change' : tech_change_for_shp_45_2099_con})
tech_change_for_shp_45_2090_int_df = pd.DataFrame({'NAME' : ca.NAME, 'yield_change' : tech_change_for_shp_45_2099_int})



ca_merge_rcp45_2099 =  ca.merge(yield_change_for_shp_45_2099_df, on = 'NAME')
ca_merge_rcp85_2099 =  ca.merge(yield_change_for_shp_85_2099_df, on = 'NAME')
ca_merge_rcp45_2099_tech_con =  ca.merge(tech_change_for_shp_45_2090_con_df, on = 'NAME')
ca_merge_rcp45_2099_tech_int =  ca.merge(tech_change_for_shp_45_2090_int_df, on = 'NAME')

#ca_merge_rcp45_2099_value =  ca.merge(yield_change_for_shp_45_2099_df_value, on = 'NAME')
#ca_merge_rcp85_2099_value =  ca.merge(yield_change_for_shp_85_2099_df_value, on = 'NAME')
df_county_yield_rcp45 = pd.DataFrame()
df_county_yield_rcp85 = pd.DataFrame()
df_county_yield_rcp45_tech_con = pd.DataFrame()
df_county_yield_rcp45_tech_int = pd.DataFrame()
for i in range(0,16):
    df_county_yield_rcp45_ind = pd.DataFrame({'County' : str(county_list[i]) , 'Yield Change % by 2099' : locals()[str(county_list[i])+'county_yield_change_2099'][:,1]})
    df_county_yield_rcp45 = pd.concat((df_county_yield_rcp45, df_county_yield_rcp45_ind))
    df_county_yield_rcp85_ind = pd.DataFrame({'County' : str(county_list[i]) , 'Yield Change % by 2099' : locals()[str(county_list[i])+'county_yield_change_2099'][:,3]})
    df_county_yield_rcp85 = pd.concat((df_county_yield_rcp85, df_county_yield_rcp85_ind))
    df_county_yield_rcp45_tech_con_ind = pd.DataFrame({'County' : str(county_list[i]) , 'Yield Change % by 2099' :  locals()[str(county_list[i])+'county_tech_change_2099'][:,0]})
    df_county_yield_rcp45_tech_con = pd.concat((df_county_yield_rcp45_tech_con, df_county_yield_rcp45_tech_con_ind))
    df_county_yield_rcp45_tech_int_ind = pd.DataFrame({'County' : str(county_list[i]) , 'Yield Change % by 2099' :  locals()[str(county_list[i])+'county_tech_change_2099'][:,1]})
    df_county_yield_rcp45_tech_int = pd.concat((df_county_yield_rcp45_tech_int, df_county_yield_rcp45_tech_int_ind))
yield_for_shp_obs_2020 = np.zeros((58))
yield_for_shp_obs_2020[:] = np.nan
for i in range(0,58):
    for index in range(0,16):
        if county_list[index] == ca.NAME[i]:
            yield_for_shp_obs_2020[i] = yield_csv[-1, 1:][index]
yield_for_shp_obs_2020_df = pd.DataFrame({'NAME' : ca.NAME, 'Observation' : yield_for_shp_obs_2020})
ca_merge_obs = ca.merge(yield_for_shp_obs_2020_df)


fig = plt.figure()
fig.set_figheight(30)
fig.set_figwidth(50)
spec = gridspec.GridSpec(nrows=110, ncols=7, width_ratios=[1,1,0,0.6,1,0,0.6], wspace = 0,hspace = 20)

ax0 = plt.subplot(spec[0:50,0])
ca_merge_obs.plot(ax = ax0, column = ca_merge_obs.Observation,edgecolor='black',missing_kwds={'color': 'grey'}, legend = True, cmap = 'Greens',legend_kwds={'orientation': "horizontal", 'aspect' : 10}, vmin = 0.5, vmax = 2)
ax0.set_axis_off()
ax0.set_title('Observed yield at 2020 (ton/acre)', fontsize = 35, y = 1.08)
ca_county_remove_shp['coords'] = ca_county_remove_shp['geometry'].apply(lambda x: x.representative_point().coords[:])
ca_county_remove_shp['coords'] = [coords[0] for coords in ca_county_remove_shp['coords']]
county_order_N_S = ['Tehama', 'Butte', 'Glenn', 'Yuba', 'Colusa', 'Sutter', 'Yolo', 'Solano', 'San Joaquin', 'Stanislaus', 'Madera', 'Merced', 'Fresno', 'Tulare', 'Kings', 'Kern']
ticks_county_order_N_S = ['[1]', '[2]', '[3]', '[4]', '[5]', '[6]', '[7]', '[8]', '[9]', '[10]', '[11]', '[12]', '[13]', '[14]', '[15]', '[16]']
#for idx, row in ca_county_remove_shp.iterrows():
 #  plt.annotate(row['N_S_order'], xy=(row['coords'][0],row['coords'][1]-20000) , horizontalalignment='center', color='black', fontsize =30)
#for idx in range(16):

ax1 = plt.subplot(spec[0:50,1])
ca_merge_rcp45_2099_tech_con.plot(ax = ax1, column = ca_merge_rcp45_2099_tech_int.yield_change, edgecolor='black',missing_kwds={'color': 'grey'}, legend = True, cmap = 'Purples',legend_kwds={'orientation': "horizontal", 'ticks': [-50, 0, 100, 200, 300]}, vmin = -50, vmax = 300)
fig = ax1.figure
cb_ax = fig.axes[3]
cb_ax.tick_params(labelsize = 35)
#cb_ax.set_title("Yield change, %", fontsize=35, y=-2.5)
cb_ax.set_position([0.505, 0.19, 0.2, 0.4])
cb_ax = fig.axes[1]
cb_ax.tick_params(labelsize = 35)
#cb_ax.set_title("Yield change, %", fontsize=35, y=-2.5)
cb_ax.set_position([0.18, 0.19, 0.1, 0.4])
ax1.set_axis_off()
idx_county_order_N_S = ['[1]Tehama', '[2]Butte', '[3]Glenn', '[4]Yuba', '[5]Colusa', '[6]Sutter', '[7]Yolo', '[8]Solano', '[9]San Joaquin', '[10]Stanislaus', '[11]Madera', '[12]Merced', '[13]Fresno', '[14]Tulare', '[15]Kings', '[16]Kern']
for idx in range(8):
    ax1.text(x = -15400000, y = (2500000-100000*idx), s=idx_county_order_N_S[idx], fontsize =35)
for idx in range(8,16):
    ax1.text(x = -14800000, y = (2500000-100000*(idx-8)), s=idx_county_order_N_S[idx], fontsize =35)
ax1_box = plt.subplot(spec[0:36,3])
norm = matplotlib.colors.Normalize(vmax = 200,vmin = -50)
my_pal = {'Butte' : cm.Purples(norm(median_tech_change_2099[0,1])), 'Colusa': cm.Purples(norm(median_tech_change_2099[1,1])), 'Fresno' : cm.Purples(norm(median_tech_change_2099[2,1])), 'Glenn' : cm.Purples(norm(median_tech_change_2099[3,1])),
          'Kern' : cm.Purples(norm(median_tech_change_2099[4,1])), 'Kings' : cm.Purples(norm(median_tech_change_2099[5,1])), 'Madera' : cm.Purples(norm(median_tech_change_2099[6,1])), 'Merced' : cm.Purples(norm(median_tech_change_2099[7,1])),
          'San Joaquin' : cm.Purples(norm(median_tech_change_2099[8,1])), 'Solano' : cm.Purples(norm(median_tech_change_2099[9,1])), 'Stanislaus' : cm.Purples(norm(median_tech_change_2099[10,1])), 'Sutter' : cm.Purples(norm(median_tech_change_2099[11,1])),
          'Tehama' :cm.Purples(norm(median_tech_change_2099[12,1])), 'Tulare' : cm.Purples(norm(median_tech_change_2099[13,1])), 'Yolo' : cm.Purples(norm(median_tech_change_2099[14,1])), 'Yuba': cm.Purples(norm(median_tech_change_2099[15,1]))}
sns.boxplot(ax = ax1_box,x = 'Yield Change % by 2099', y = 'County', data = df_county_yield_rcp45_tech_int,  order = county_order_N_S, palette = my_pal, showfliers = False)
ax1_box.text(-160, -2, 'Yield change from 2020 due to technological improvement (%)', fontsize = 35)
ax1_box.set_xticks([-50, 0, 100, 200, 300])
ax1_box.set_xticklabels(['-50', '0', '100', '200','300'])
#plt.xlabel('Yield Change %', fontsize = 35)
plt.xlabel('')
plt.ylabel('')
plt.xticks(fontsize = 25)
plt.yticks(np.arange(0,16), ticks_county_order_N_S, fontsize = 25)
plt.xlim(-50,300)
plt.axvline(x=-100, linestyle = 'dashed', color = 'r')
plt.axvline(x=0, linestyle = 'dashed', color = 'r')
ax1_box.spines['top'].set_visible(False)
ax1_box.spines['right'].set_visible(False)


ax2 = plt.subplot(spec[0:35,4])
ca_merge_rcp45_2099_tech_int.plot(ax = ax2, column = ca_merge_rcp45_2099_tech_con.yield_change, edgecolor='black',missing_kwds={'color': 'grey'}, legend = False, cmap = 'Purples', vmin = -50, vmax = 300)
#cb_ax.tick_params(labelsize = 35)
#cb_ax.set_title("Yield change, %", fontsize=35)
#cb_ax.set_position([0.5, 0.145, 0.2, 0.4])
ax2.set_axis_off()

ax2_box = plt.subplot(spec[0:36,6])
norm = matplotlib.colors.Normalize(vmax = 200,vmin = -50)
my_pal = {'Butte' : cm.Purples(norm(median_tech_change_2099[0,0])), 'Colusa': cm.Purples(norm(median_tech_change_2099[1,0])), 'Fresno' : cm.Purples(norm(median_tech_change_2099[2,0])), 'Glenn' : cm.Purples(norm(median_tech_change_2099[3,0])),
          'Kern' : cm.Purples(norm(median_tech_change_2099[4,0])), 'Kings' : cm.Purples(norm(median_tech_change_2099[5,0])), 'Madera' : cm.Purples(norm(median_tech_change_2099[6,0])), 'Merced' : cm.Purples(norm(median_tech_change_2099[7,0])),
          'San Joaquin' : cm.Purples(norm(median_tech_change_2099[8,0])), 'Solano' : cm.Purples(norm(median_tech_change_2099[9,0])), 'Stanislaus' : cm.Purples(norm(median_tech_change_2099[10,0])), 'Sutter' : cm.Purples(norm(median_tech_change_2099[11,0])),
          'Tehama' :cm.Purples(norm(median_tech_change_2099[12,0])), 'Tulare' : cm.Purples(norm(median_tech_change_2099[13,0])), 'Yolo' : cm.Purples(norm(median_tech_change_2099[14,0])), 'Yuba': cm.Purples(norm(median_tech_change_2099[15,0]))}
sns.boxplot(ax = ax2_box,x = 'Yield Change % by 2099', y = 'County', data = df_county_yield_rcp45_tech_con,  order = county_order_N_S, palette = my_pal, showfliers = False)
ax2_box.set_xticks([-50, 0, 100, 200, 300])
ax2_box.set_xticklabels(['-50', '0', '100', '200','300'])
plt.xlabel('')
plt.ylabel('')
plt.xticks(fontsize = 25)
plt.yticks(np.arange(0,16), ticks_county_order_N_S, fontsize = 25)
plt.xlim(-50,300)
plt.axvline(x=-100, linestyle = 'dashed', color = 'r')
plt.axvline(x=0, linestyle = 'dashed', color = 'r')
ax2_box.spines['top'].set_visible(False)
ax2_box.spines['right'].set_visible(False)


ax3 = plt.subplot(spec[60:110,1])
ca_merge_rcp45_2099.plot(ax = ax3, column = ca_merge_rcp45_2099.rcp45_2099,edgecolor='black',missing_kwds={'color': 'grey'}, legend = True, cmap = 'OrRd_r', figsize = (15,15),vmin = -100, vmax = 0,legend_kwds={'orientation': "horizontal", 'ticks': [-100, -50, 0]})
cb_ax = fig.axes[8]
cb_ax.tick_params(labelsize = 35)
#cb_ax.set_title("Yield change, %", fontsize=35, y=-2.5)
cb_ax.set_position([0.505, -0.198, 0.2, 0.4])
ax3.set_axis_off()

ax3_box = plt.subplot(spec[60:96,3])
norm = matplotlib.colors.Normalize(vmax = 0,vmin = -100)
my_pal = {'Butte' : cm.OrRd_r(norm(median_yield_change_2099[0,1])), 'Colusa': cm.OrRd_r(norm(median_yield_change_2099[1,1])), 'Fresno' : cm.OrRd_r(norm(median_yield_change_2099[2,1])), 'Glenn' : cm.OrRd_r(norm(median_yield_change_2099[3,1])),
          'Kern' : cm.OrRd_r(norm(median_yield_change_2099[4,1])), 'Kings' : cm.OrRd_r(norm(median_yield_change_2099[5,1])), 'Madera' : cm.OrRd_r(norm(median_yield_change_2099[6,1])), 'Merced' : cm.OrRd_r(norm(median_yield_change_2099[7,1])),
          'San Joaquin' : cm.OrRd_r(norm(median_yield_change_2099[8,1])), 'Solano' : cm.OrRd_r(norm(median_yield_change_2099[9,1])), 'Stanislaus' : cm.OrRd_r(norm(median_yield_change_2099[10,1])), 'Sutter' : cm.OrRd_r(norm(median_yield_change_2099[11,1])),
          'Tehama' :cm.OrRd_r(norm(median_yield_change_2099[12,1])), 'Tulare' : cm.OrRd_r(norm(median_yield_change_2099[13,1])), 'Yolo' : cm.OrRd_r(norm(median_yield_change_2099[14,1])), 'Yuba': cm.OrRd_r(norm(median_yield_change_2099[15,1]))}
sns.boxplot(ax = ax3_box, x = 'Yield Change % by 2099', y = 'County', data = df_county_yield_rcp45,  order = county_order_N_S, palette = my_pal, showfliers = False)
ax3_box.text(-100, -3, 'Yield change from 2020 due to climate (%)', fontsize = 35)
ax3_box.text(-200, -1, 'RCP4.5', fontsize = 35)
ax3_box.set_xticks([-100, -50, 0, 50])
ax3_box.set_xticklabels(['-100', '-50', '0', '50'])
plt.xlabel('', fontsize = 35)
plt.ylabel('')
plt.xticks(fontsize = 25)
plt.yticks(np.arange(0,16), ticks_county_order_N_S, fontsize = 25)
plt.xlim(-105,50)
plt.axvline(x=-100, linestyle = 'dashed', color = 'r')
plt.axvline(x=0, linestyle = 'dashed', color = 'r')
ax3_box.spines['top'].set_visible(False)
ax3_box.spines['right'].set_visible(False)


ax4 = plt.subplot(spec[60:95,4])
ca_merge_rcp85_2099.plot(ax = ax4, column = ca_merge_rcp85_2099.rcp85_2099,edgecolor='black',missing_kwds={'color': 'grey'}, legend = False, cmap = 'OrRd_r', figsize = (15,15),vmin = -100, vmax = 0)
ax4.set_axis_off()
ax4_box = plt.subplot(spec[60:96,6])
my_pal = {'Butte' : cm.OrRd_r(norm(median_yield_change_2099[0,3])), 'Colusa': cm.OrRd_r(norm(median_yield_change_2099[1,3])), 'Fresno' : cm.OrRd_r(norm(median_yield_change_2099[2,3])), 'Glenn' : cm.OrRd_r(norm(median_yield_change_2099[3,3])),
          'Kern' : cm.OrRd_r(norm(median_yield_change_2099[4,3])), 'Kings' : cm.OrRd_r(norm(median_yield_change_2099[5,3])), 'Madera' : cm.OrRd_r(norm(median_yield_change_2099[6,3])), 'Merced' : cm.OrRd_r(norm(median_yield_change_2099[7,3])),
          'San Joaquin' : cm.OrRd_r(norm(median_yield_change_2099[8,3])), 'Solano' : cm.OrRd_r(norm(median_yield_change_2099[9,3])), 'Stanislaus' : cm.OrRd_r(norm(median_yield_change_2099[10,3])), 'Sutter' : cm.OrRd_r(norm(median_yield_change_2099[11,3])),
          'Tehama' :cm.OrRd_r(norm(median_yield_change_2099[12,3])), 'Tulare' : cm.OrRd_r(norm(median_yield_change_2099[13,3])), 'Yolo' : cm.OrRd_r(norm(median_yield_change_2099[14,3])), 'Yuba': cm.OrRd_r(norm(median_yield_change_2099[15,3]))}
sns.boxplot(ax = ax4_box, x = 'Yield Change % by 2099', y = 'County', data = df_county_yield_rcp85,  order = county_order_N_S, palette = my_pal, showfliers = False)
ax4_box.text(-200, -1, 'RCP8.5', fontsize = 35)
ax4_box.set_xticks([-100, -50, 0, 50])
ax4_box.set_xticklabels(['-100', '-50', '0', '50'])
plt.xlabel('', fontsize = 35)
plt.ylabel('')
plt.xticks(fontsize = 25)
plt.yticks(np.arange(0,16), ticks_county_order_N_S, fontsize = 25)
plt.xlim(-105,50)
plt.axvline(x=-100, linestyle = 'dashed', color = 'r')
plt.axvline(x=0, linestyle = 'dashed', color = 'r')
ax4_box.spines['top'].set_visible(False)
ax4_box.spines['right'].set_visible(False)
plt.savefig(save_path+'map.pdf', dpi = 300, bbox_inches='tight')



## Figure 3: Uncertainty

def poly_transform(X,Y):
    poly_array = np.zeros((Y.shape[0],Y.shape[1]))
    Y_len = Y.shape[1]
    for i in range(Y_len):
        Poly_fit = np.polyfit(X, Y[:,i], deg = 4)
        Poly_func = np.poly1d(Poly_fit)
        Poly_pred = Poly_func(X)
        poly_array[:,i] = Poly_pred
    return(poly_array)

yield_all_sum_rcp45 = np.row_stack((yield_all_hist_rcp45, yield_all_future_rcp45))
yield_all_sum_rcp45_s = np.row_stack((yield_all_hist_rcp45_s, yield_all_future_rcp45_s))
yield_all_sum_rcp45_m = np.row_stack((yield_all_hist_rcp45_m, yield_all_future_rcp45_m))
yield_all_sum_rcp85 = np.row_stack((yield_all_hist_rcp85, yield_all_future_rcp85))
yield_all_sum_rcp85_s = np.row_stack((yield_all_hist_rcp85_s, yield_all_future_rcp85_s))
yield_all_sum_rcp85_m = np.row_stack((yield_all_hist_rcp85_m, yield_all_future_rcp85_m))



poly_X = np.arange(1980,2100)
yield_all_sum_rcp45_s_poly = poly_transform(poly_X, yield_all_sum_rcp45_s).reshape(120,18,1000)[40:120]
yield_all_sum_rcp85_s_poly = poly_transform(poly_X, yield_all_sum_rcp85_s).reshape(120,18,1000)[40:120]
yield_all_sum_rcp45_m_poly = poly_transform(poly_X, yield_all_sum_rcp45_m).reshape(120,18,1000)[40:120]
yield_all_sum_rcp85_m_poly = poly_transform(poly_X, yield_all_sum_rcp85_m).reshape(120,18,1000)[40:120]
yield_all_sum_rcp45_poly = poly_transform(poly_X, yield_all_sum_rcp45).reshape(120,18,1000)[40:120]
yield_all_sum_rcp85_poly = poly_transform(poly_X, yield_all_sum_rcp85).reshape(120,18,1000)[40:120]

yield_all_sum_rcp45_2020_2099 = yield_all_sum_rcp45[40:120]
yield_all_sum_rcp45_s_2020_2099 =  yield_all_sum_rcp45_s[40:120]
yield_all_sum_rcp45_m_2020_2099 =  yield_all_sum_rcp45_m[40:120]
yield_all_sum_rcp85_2020_2099 =  yield_all_sum_rcp85[40:120]
yield_all_sum_rcp85_s_2020_2099=  yield_all_sum_rcp85_s[40:120]
yield_all_sum_rcp85_m_2020_2099=  yield_all_sum_rcp85_m[40:120]

num_rcp = 2
num_stat_model = 1000
num_tech = 3
num_clim_model = 18

## calculate climate model uncertainty MC
MC_rcp45 = np.zeros(80)
MC_rcp45_s = np.zeros(80)
MC_rcp45_m = np.zeros(80)
MC_rcp85 = np.zeros(80)
MC_rcp85_s = np.zeros(80) 
MC_rcp85_m = np.zeros(80) 
for year in range(0,80):
    MC_rcp45[year] = np.var(np.median(yield_all_sum_rcp45_poly, axis = 2)[year,:])
    MC_rcp45_s[year] = np.var(np.median(yield_all_sum_rcp45_s_poly, axis = 2)[year,:])
    MC_rcp45_m[year] = np.var(np.median(yield_all_sum_rcp45_m_poly, axis = 2)[year,:])
    MC_rcp85[year] = np.var(np.median(yield_all_sum_rcp85_poly, axis = 2)[year,:])
    MC_rcp85_s[year] = np.var(np.median(yield_all_sum_rcp85_s_poly, axis = 2)[year,:])
    MC_rcp85_m[year] = np.var(np.median(yield_all_sum_rcp85_m_poly, axis = 2)[year,:])   
    
MC_time_series = (MC_rcp45+MC_rcp45_s+MC_rcp45_m+MC_rcp85+MC_rcp85_s+MC_rcp85_m)/6

## calculate stat model uncertainty MS
MS_rcp45 = np.zeros(80)
MS_rcp45_s = np.zeros(80)
MS_rcp45_m = np.zeros(80)
MS_rcp85 = np.zeros(80)
MS_rcp85_s = np.zeros(80) 
MS_rcp85_m = np.zeros(80) 
for year in range(0,80):
        MS_rcp45[year] = np.var(np.mean(yield_all_sum_rcp45_poly, axis = 1)[year,:])
        MS_rcp45_s[year] = np.var(np.mean(yield_all_sum_rcp45_s_poly, axis = 1)[year,:])
        MS_rcp45_m[year] = np.var(np.mean(yield_all_sum_rcp45_m_poly, axis = 1)[year,:])
        MS_rcp85[year] = np.var(np.mean(yield_all_sum_rcp85_poly, axis = 1)[year,:])  
        MS_rcp85_s[year] = np.var(np.mean(yield_all_sum_rcp85_s_poly, axis = 1)[year,:])
        MS_rcp85_m[year] = np.var(np.mean(yield_all_sum_rcp85_m_poly, axis = 1)[year,:])
MS_time_series = (MS_rcp45 + MS_rcp45_s + MS_rcp45_m + MS_rcp85 + MS_rcp85_s + MS_rcp85_m)/6


## calculate tech trend scenario uncertainty ST
yield_all_sum_rcp45_tech_scenario = np.zeros((80,18,1000,3))    
yield_all_sum_rcp45_tech_scenario[:,:,:,0] = yield_all_sum_rcp45_poly
yield_all_sum_rcp45_tech_scenario[:,:,:,1] = yield_all_sum_rcp45_s_poly
yield_all_sum_rcp45_tech_scenario[:,:,:,2] = yield_all_sum_rcp45_m_poly
yield_all_sum_rcp85_tech_scenario = np.zeros((80,18,1000,3))    
yield_all_sum_rcp85_tech_scenario[:,:,:,0] = yield_all_sum_rcp85_poly
yield_all_sum_rcp85_tech_scenario[:,:,:,1] = yield_all_sum_rcp85_s_poly
yield_all_sum_rcp85_tech_scenario[:,:,:,2] = yield_all_sum_rcp85_m_poly

ST_rcp45 = np.zeros(80)
ST_rcp85 = np.zeros(80)
for year in range(0,80):
    ST_rcp45[year] = np.var(np.median(np.mean(yield_all_sum_rcp45_tech_scenario, axis = 1), axis = 1)[year,:])
    ST_rcp85[year] = np.var(np.median(np.mean(yield_all_sum_rcp85_tech_scenario, axis = 1), axis = 1)[year,:])
ST_time_series = ((ST_rcp45 + ST_rcp85)/2)

## calculate rcp scenario uncertainty SR
yield_all_sum_rcp_scenario_tech = np.zeros((80,18,1000,2))    
yield_all_sum_rcp_scenario_tech[:,:,:,0] = yield_all_sum_rcp45_poly
yield_all_sum_rcp_scenario_tech[:,:,:,1] = yield_all_sum_rcp85_poly
yield_all_sum_rcp_scenario_no_tech = np.zeros((80,18,1000,2))    
yield_all_sum_rcp_scenario_no_tech[:,:,:,0] = yield_all_sum_rcp45_s_poly
yield_all_sum_rcp_scenario_no_tech[:,:,:,1] = yield_all_sum_rcp85_s_poly
yield_all_sum_rcp_scenario_int_tech = np.zeros((80,18,1000,2))    
yield_all_sum_rcp_scenario_int_tech[:,:,:,0] = yield_all_sum_rcp45_m_poly
yield_all_sum_rcp_scenario_int_tech[:,:,:,1] = yield_all_sum_rcp85_m_poly

SR_tech = np.zeros(80)
SR_no_tech = np.zeros(80)
SR_int_tech = np.zeros(80)
for year in range(0,80):
    SR_tech[year] = np.var(np.median(np.mean(yield_all_sum_rcp_scenario_tech, axis = 1), axis = 1)[year,:])
    SR_no_tech[year] = np.var(np.median(np.mean(yield_all_sum_rcp_scenario_no_tech, axis = 1), axis = 1)[year,:])
    SR_int_tech[year] = np.var(np.median(np.mean(yield_all_sum_rcp_scenario_int_tech, axis = 1), axis = 1)[year,:])
SR_time_series = ((SR_tech + SR_no_tech + SR_int_tech)/3)


## calculate internal 
Intvar_rcp45 = np.zeros(80)
Intvar_rcp45_s = np.zeros(80)
Intvar_rcp45_m = np.zeros(80)
Intvar_rcp85 = np.zeros(80)
Intvar_rcp85_s = np.zeros(80)
Intvar_rcp85_m = np.zeros(80)
residual_rcp45 = np.zeros((80,18,1000))
residual_rcp45_s = np.zeros((80,18,1000))
residual_rcp45_m = np.zeros((80,18,1000))
residual_rcp85 = np.zeros((80,18,1000))
residual_rcp85_s = np.zeros((80,18,1000))
residual_rcp85_m = np.zeros((80,18,1000))

for trial in range(0,1000):
    for climate in range(0,18):
        residual_rcp45[:,climate,trial] = yield_all_sum_rcp45_poly[:,climate,trial]-np.array(pd.DataFrame(yield_all_sum_rcp45_2020_2099.reshape(80,18,1000)[:,climate,trial]).rolling(10,min_periods=5, center = True).mean()).reshape(80)
        residual_rcp45_s[:,climate,trial] =  yield_all_sum_rcp45_s_poly[:,climate,trial]-np.array(pd.DataFrame(yield_all_sum_rcp45_s_2020_2099.reshape(80,18,1000)[:,climate,trial]).rolling(10,min_periods=5, center = True).mean()).reshape(80)
        residual_rcp45_m[:,climate,trial] =  yield_all_sum_rcp45_m_poly[:,climate,trial]-np.array(pd.DataFrame(yield_all_sum_rcp45_m_2020_2099.reshape(80,18,1000)[:,climate,trial]).rolling(10,min_periods=5, center = True).mean()).reshape(80)
        residual_rcp85[:,climate,trial] =  yield_all_sum_rcp85_poly[:,climate,trial]-np.array(pd.DataFrame(yield_all_sum_rcp85_2020_2099.reshape(80,18,1000)[:,climate,trial]).rolling(10,min_periods=5, center = True).mean()).reshape(80)
        residual_rcp85_s[:,climate,trial] =  yield_all_sum_rcp85_s_poly[:,climate,trial]-np.array(pd.DataFrame(yield_all_sum_rcp85_s_2020_2099.reshape(80,18,1000)[:,climate,trial]).rolling(10,min_periods=5, center = True).mean()).reshape(80)
        residual_rcp85_m[:,climate,trial] =  yield_all_sum_rcp85_m_poly[:,climate,trial]-np.array(pd.DataFrame(yield_all_sum_rcp85_m_2020_2099.reshape(80,18,1000)[:,climate,trial]).rolling(10,min_periods=5, center = True).mean()).reshape(80)


for year in range(0,80):
    Intvar_rcp45[year] = np.var(residual_rcp45[year,:,:])
    Intvar_rcp45_s[year] = np.var(residual_rcp45_s[year,:,:])
    Intvar_rcp45_m[year] = np.var(residual_rcp45_m[year,:,:])
    Intvar_rcp85[year] = np.var(residual_rcp85[year,:,:])
    Intvar_rcp85_s[year] = np.var(residual_rcp85_s[year,:,:])
    Intvar_rcp85_m[year] = np.var(residual_rcp85_m[year,:,:])
Intvar_time_series = ((Intvar_rcp45+Intvar_rcp45_s+Intvar_rcp45_m+Intvar_rcp85+Intvar_rcp85_s+Intvar_rcp85_m)/6)

#####calculate mean/median yield + uncertainty
mean_median_yield_2020_2099_rcp45 = np.median(np.mean(yield_all_sum_rcp45_2020_2099.reshape(80,18,1000),axis=1),axis=1)
mean_median_yield_2020_2099_rcp45_s =  np.median(np.mean(yield_all_sum_rcp45_s_2020_2099.reshape(80,18,1000),axis=1),axis=1)
mean_median_yield_2020_2099_rcp45_m =  np.median(np.mean(yield_all_sum_rcp45_m_2020_2099.reshape(80,18,1000),axis=1),axis=1)
mean_median_yield_2020_2099_rcp85 =  np.median(np.mean(yield_all_sum_rcp85_2020_2099.reshape(80,18,1000),axis=1),axis=1)
mean_median_yield_2020_2099_rcp85_s =  np.median(np.mean(yield_all_sum_rcp85_s_2020_2099.reshape(80,18,1000),axis=1),axis=1)
mean_median_yield_2020_2099_rcp85_m =  np.median(np.mean(yield_all_sum_rcp85_m_2020_2099.reshape(80,18,1000),axis=1),axis=1)

mean_median_yield_2020_2099_rcp45_10_yr_running_mean = pd.DataFrame(mean_median_yield_2020_2099_rcp45).rolling(10,min_periods=5, center = True).mean()
mean_median_yield_2020_2099_rcp45_s_10_yr_running_mean = pd.DataFrame(mean_median_yield_2020_2099_rcp45_s).rolling(10,min_periods=5, center = True).mean()
mean_median_yield_2020_2099_rcp45_m_10_yr_running_mean = pd.DataFrame(mean_median_yield_2020_2099_rcp45_m).rolling(10,min_periods=5, center = True).mean()

mean_median_yield_2020_2099_rcp85_10_yr_running_mean = pd.DataFrame(mean_median_yield_2020_2099_rcp85).rolling(10,min_periods=5, center = True).mean()
mean_median_yield_2020_2099_rcp85_s_10_yr_running_mean = pd.DataFrame(mean_median_yield_2020_2099_rcp85_s).rolling(10,min_periods=5, center = True).mean()
mean_median_yield_2020_2099_rcp85_m_10_yr_running_mean = pd.DataFrame(mean_median_yield_2020_2099_rcp85_m).rolling(10,min_periods=5, center = True).mean()


mean_median_yield_2020_2099 = np.array((mean_median_yield_2020_2099_rcp45_10_yr_running_mean + mean_median_yield_2020_2099_rcp45_s_10_yr_running_mean + mean_median_yield_2020_2099_rcp45_m_10_yr_running_mean + mean_median_yield_2020_2099_rcp85_10_yr_running_mean
                               + mean_median_yield_2020_2099_rcp85_s_10_yr_running_mean + mean_median_yield_2020_2099_rcp85_m_10_yr_running_mean)/6).reshape(80)

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
MC_rcp45 = np.zeros(80)
MC_rcp85 = np.zeros(80)
for year in range(0,80):
    MC_rcp45[year] = np.var(np.median(yield_all_sum_rcp45_poly, axis = 2)[year,:])
    MC_rcp85[year] = np.var(np.median(yield_all_sum_rcp85_poly, axis = 2)[year,:])
        
MC_time_series_tech = (MC_rcp45+MC_rcp85)/2

## calculate stat model uncertainty MS
MS_rcp45 = np.zeros(80)
MS_rcp85 = np.zeros(80)
for year in range(0,80):
        MS_rcp45[year] = np.var(np.mean(yield_all_sum_rcp45_poly, axis = 1)[year,:])
        MS_rcp85[year] = np.var(np.mean(yield_all_sum_rcp85_poly, axis = 1)[year,:])  
MS_time_series_tech = (MS_rcp45 + MS_rcp85)/2




## calculate rcp scenario uncertainty SR
yield_all_sum_rcp_scenario_tech = np.zeros((80,18,1000,2))    
yield_all_sum_rcp_scenario_tech[:,:,:,0] = yield_all_sum_rcp45_poly
yield_all_sum_rcp_scenario_tech[:,:,:,1] = yield_all_sum_rcp85_poly


SR_tech = np.zeros(80)
for year in range(0,80):
    SR_tech[year] = np.var(np.median(np.mean(yield_all_sum_rcp_scenario_tech, axis = 1), axis = 1)[year,:])
SR_time_series_tech = SR_tech


## calculate internal 
Intvar_rcp45 = np.zeros(80)
Intvar_rcp85 = np.zeros(80)
residual_rcp45 = np.zeros((80,18,1000))
residual_rcp85 = np.zeros((80,18,1000))


for trial in range(0,1000):
    for climate in range(0,18):
        residual_rcp45[:,climate,trial] = yield_all_sum_rcp45_poly[:,climate,trial]-np.array(pd.DataFrame(yield_all_sum_rcp45_2020_2099.reshape(80,18,1000)[:,climate,trial]).rolling(10,min_periods=5, center = True).mean()).reshape(80)
        residual_rcp85[:,climate,trial] =  yield_all_sum_rcp85_poly[:,climate,trial]-np.array(pd.DataFrame(yield_all_sum_rcp85_2020_2099.reshape(80,18,1000)[:,climate,trial]).rolling(10,min_periods=5, center = True).mean()).reshape(80)


for year in range(0,80):
    Intvar_rcp45[year] = np.var(residual_rcp45[year,:,:])
    Intvar_rcp85[year] = np.var(residual_rcp85[year,:,:])
Intvar_time_series_tech = ((Intvar_rcp45+Intvar_rcp85)/2)

mean_median_yield_2020_2099_tech = np.array((mean_median_yield_2020_2099_rcp45_10_yr_running_mean  + mean_median_yield_2020_2099_rcp85_10_yr_running_mean)/2).reshape(80)

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
MC_rcp45_s = np.zeros(80)
MC_rcp85_s = np.zeros(80) 
for year in range(0,80):
    MC_rcp45_s[year] = np.var(np.median(yield_all_sum_rcp45_s_poly, axis = 2)[year,:])
    MC_rcp85_s[year] = np.var(np.median(yield_all_sum_rcp85_s_poly, axis = 2)[year,:])
        
MC_time_series_no_tech = (MC_rcp45_s+MC_rcp85_s)/2

## calculate stat model uncertainty MS
MS_rcp45_s = np.zeros(80)
MS_rcp85_s = np.zeros(80) 
for year in range(0,80):
        MS_rcp45_s[year] = np.var(np.mean(yield_all_sum_rcp45_s_poly, axis = 1)[year,:])
        MS_rcp85_s[year] = np.var(np.mean(yield_all_sum_rcp85_s_poly, axis = 1)[year,:])
MS_time_series_no_tech = (MS_rcp45_s + MS_rcp85_s)/2



## calculate rcp scenario uncertainty SR
yield_all_sum_rcp_scenario_no_tech = np.zeros((80,18,1000,2))    
yield_all_sum_rcp_scenario_no_tech[:,:,:,0] = yield_all_sum_rcp45_s_poly
yield_all_sum_rcp_scenario_no_tech[:,:,:,1] = yield_all_sum_rcp85_s_poly

SR_no_tech = np.zeros(80)
for year in range(0,80):
    SR_no_tech[year] = np.var(np.median(np.mean(yield_all_sum_rcp_scenario_no_tech, axis = 1), axis = 1)[year,:])
SR_time_series_no_tech = SR_no_tech


## calculate internal 
Intvar_rcp45_s = np.zeros(80)
Intvar_rcp85_s = np.zeros(80)
residual_rcp45_s = np.zeros((80,18,1000))
residual_rcp85_s = np.zeros((80,18,1000))


for trial in range(0,1000):
    for climate in range(0,18):
        residual_rcp45_s[:,climate,trial] =  yield_all_sum_rcp45_s_poly[:,climate,trial]-np.array(pd.DataFrame(yield_all_sum_rcp45_s_2020_2099.reshape(80,18,1000)[:,climate,trial]).rolling(10,min_periods=5, center = True).mean()).reshape(80)
        residual_rcp85_s[:,climate,trial] =  yield_all_sum_rcp85_s_poly[:,climate,trial]-np.array(pd.DataFrame(yield_all_sum_rcp85_s_2020_2099.reshape(80,18,1000)[:,climate,trial]).rolling(10,min_periods=5, center = True).mean()).reshape(80)


for year in range(0,80):
    Intvar_rcp45_s[year] = np.var(residual_rcp45_s[year,:,:])
    Intvar_rcp85_s[year] = np.var(residual_rcp85_s[year,:,:])
Intvar_time_series_no_tech = ((Intvar_rcp45_s+Intvar_rcp85_s)/2)

mean_median_yield_2020_2099_no_tech = np.array((mean_median_yield_2020_2099_rcp45_s_10_yr_running_mean  + mean_median_yield_2020_2099_rcp85_s_10_yr_running_mean)/2).reshape(80)


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

###plot
plt.figure(figsize = (45,13))
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
              ,Var_total_10yr_mean[:,3],Var_total_10yr_mean[:,4],labels = ['Climate Model', 'Stats Model', 'Tech Scenario','RCP Scenario','Internal'], colors = ['354AA1', '85B1D4','lightgreen', '007F3C', 'FF6E04'])
plt.ylim(0,100)
plt.xlim(2020,2100)
plt.xticks(fontsize =35, rotation = 35)
plt.yticks(np.linspace(20,100,5).astype(int),np.linspace(20,100,5).astype(int),fontsize = 35)
plt.title('Fractional contribution to total uncertainty (%) \n tech-trend scenario included' , fontsize = 35, y = 1.02)
plt.xlabel('Year', fontsize = 35, labelpad = 20)

plt.subplot(1,3,1)
plt.fill_between(np.arange(2020,2100), y1 = y_upper_tech_no_tech[0,:], y2 = y_lower_tech_no_tech[0,:],color = '#FF6E04',  label = 'Int. Variability')
plt.fill_between(np.arange(2020,2100), y1 = y_upper_tech_no_tech[1,:], y2 = y_lower_tech_no_tech[1,:],color = '#85B1D4',  label = 'Stats. Model')
plt.fill_between(np.arange(2020,2100), y1 = y_upper_tech_no_tech[2,:], y2 = y_lower_tech_no_tech[2,:],color = '#354AA1', label = 'Climate Model')
plt.fill_between(np.arange(2020,2100), y1 = y_upper_tech_no_tech[3,:], y2 = y_lower_tech_no_tech[3,:],color = '#007F3C',  label = 'RCP Scenario')
plt.fill_between(np.arange(2020,2100), y1 = y_upper_tech_no_tech[4,:], y2 = y_lower_tech_no_tech[4,:],color = 'lightgreen',  label = 'Tech-trend Scenario')
plt.xticks(fontsize =35, rotation = 35)
plt.yticks(np.linspace(0.5,2.5,5), np.linspace(0.5,2.5,5),fontsize =35)
plt.xlim(2020,2100)
plt.title('Source of uncertainty (Yield, ton/acre)', fontsize = 35, y = 1.06)
plt.legend(loc='upper right', bbox_to_anchor=(3.15, -0.2), ncol = 5, fontsize = 35, edgecolor = 'white')
plt.ylim(0,2.6)

plt.subplot(1,3,3)
Var_total_10yr_mean = np.zeros((80,4))
Var_total_10yr_mean[:,0] = MC_time_series_no_tech
Var_total_10yr_mean[:,1] = MS_time_series_no_tech
Var_total_10yr_mean[:,2] = SR_time_series_no_tech
Var_total_10yr_mean[:,3] = Intvar_time_series_no_tech
for year in range(0,80):
    Var_total_10yr_mean[year,:] = Var_total_10yr_mean[year,:] * 100 / np.sum(Var_total_10yr_mean[year,:])
plt.stackplot(np.arange(2020,2100), Var_total_10yr_mean[:,0],Var_total_10yr_mean[:,1],Var_total_10yr_mean[:,2]
              ,Var_total_10yr_mean[:,3], labels = ['Climate Model', 'Stats Model', 'RCP Scenario','Internal'], colors = ['354AA1', '85B1D4', '007F3C', 'FF6E04'])
plt.ylim(0,100)
plt.xlim(2020,2100)
plt.xticks(fontsize =35, rotation = 35)
plt.yticks(np.linspace(20,100,5).astype(int),np.linspace(20,100,5).astype(int),fontsize = 35)
plt.title('Fractional contribution to total uncertainty (%)' , fontsize = 35, y = 1.06)

plt.savefig(save_path+'uncertainty.pdf', bbox_inches='tight', dpi =300)




##Figure 4: State-level waterfall ACI analysis
ACI_list = ['Dormancy_Freeze','Dormancy_ETo','Jan_P','Bloom_P','Bloom_Tmin' ,'Bloom_ETo', 'Bloom_GDD4','Bloom_Humidity','Bloom_Windydays','Growing_ETo','Growing_GDD4', 'Growing_KDD30','Harvest_P']

aci_contribution_rcp45_total = np.load(input_path_contribution+'aci_contribution_rcp45_total.npy')
aci_contribution_rcp85_total = np.load(input_path_contribution+'aci_contribution_rcp85_total.npy')

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



fig, axs = plt.subplots(2,2,figsize=(40,30))
formatting = "{:,.1f}"
from matplotlib.ticker import FuncFormatter
index = np.array(ACI_list)[:]
data = median_rcp45_2041_2060
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
plt.plot(step.index, step.values, 'k', linewidth = 2)
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
plt.axhline(0, color='black', linewidth = 0.6, linestyle="dashed")
for i in (0,2,4,6,8,10,12):
    rect=mpatches.Rectangle([-0.5+i,-120], 1, 130, ec='white', fc='grey', alpha=0.2, clip_on=False)
    axs[0,0].add_patch(rect)
plt.tick_params(axis = 'x' , which = 'both', bottom = False, top = False, labelbottom = False)
plt.text(-3,-65, s = 'RCP4.5', fontsize = 40, rotation = 'vertical')
plt.text(4.8,20, s = '2040-2059', fontsize = 40)

formatting = "{:,.1f}"
from matplotlib.ticker import FuncFormatter
index = np.array(ACI_list)[:]
data = median_rcp45_2080_2099
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
plt.plot(step.index, step.values, 'k', linewidth = 2)
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
plt.axhline(0, color='black', linewidth = 0.6, linestyle="dashed")
for i in (0,2,4,6,8,10,12):
    rect=mpatches.Rectangle([-0.5+i,-120], 1, 130, ec='white', fc='grey', alpha=0.2, clip_on=False)
    axs[0,1].add_patch(rect)
plt.tick_params(axis = 'x' , which = 'both', bottom = False, top = False, labelbottom = False)
plt.text(4.8,20, s = '2080-2099', fontsize = 40)

formatting = "{:,.1f}"
from matplotlib.ticker import FuncFormatter
index = np.array(ACI_list)[:]
data = median_rcp85_2041_2060
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
plt.plot(step.index, step.values, 'k', linewidth = 2)
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
plt.axhline(0, color='black', linewidth = 0.6, linestyle="dashed")
for i in (0,2,4,6,8,10,12):
    rect=mpatches.Rectangle([-0.5+i,-120], 1, 130, ec='white', fc='grey', alpha=0.2, clip_on=False)
    axs[1,0].add_patch(rect)
plt.xticks(np.arange(0,len(trans)), trans.index, rotation=90, fontsize = 35)
plt.text(-3,-65, s = 'RCP8.5', fontsize = 40, rotation = 'vertical')
annotate_y = -0.5
plt.annotate('', xy = (0.05, annotate_y), xycoords = 'axes fraction', xytext = (0.225,annotate_y), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Dormancy', xy = (0.05, annotate_y-0.1), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.265, annotate_y), xycoords = 'axes fraction', xytext = (0.66,annotate_y), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Bloom', xy = (0.41,annotate_y-0.1), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.7, annotate_y), xycoords = 'axes fraction', xytext = (0.88,annotate_y), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Growth', xy = (0.73, annotate_y-0.1), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.9, annotate_y), xycoords = 'axes fraction', xytext = (0.97,annotate_y), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Harvest', xy = (0.87, annotate_y-0.1), xycoords = 'axes fraction', fontsize = 35)

formatting = "{:,.1f}"
from matplotlib.ticker import FuncFormatter
index = np.array(ACI_list)[:]
data = median_rcp85_2080_2099
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
plt.plot(step.index, step.values, 'k', linewidth = 2)
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
plt.axhline(0, color='black', linewidth = 0.6, linestyle="dashed")
for i in (0,2,4,6,8,10,12):
    rect=mpatches.Rectangle([-0.5+i,-120], 1, 130, ec='white', fc='grey', alpha=0.2, clip_on=False)
    axs[1,1].add_patch(rect)
plt.xticks(np.arange(0,len(trans)), trans.index, rotation=90, fontsize = 35)
annotate_y = -0.5
plt.annotate('', xy = (0.05, annotate_y), xycoords = 'axes fraction', xytext = (0.225,annotate_y), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Dormancy', xy = (0.05, annotate_y-0.1), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.265, annotate_y), xycoords = 'axes fraction', xytext = (0.66,annotate_y), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Bloom', xy = (0.41,annotate_y-0.1), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.7, annotate_y), xycoords = 'axes fraction', xytext = (0.88,annotate_y), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Growth', xy = (0.73, annotate_y-0.1), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.9, annotate_y), xycoords = 'axes fraction', xytext = (0.97,annotate_y), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Harvest', xy = (0.87, annotate_y-0.1), xycoords = 'axes fraction', fontsize = 35)
plt.savefig(save_path+'waterfall_state_all.pdf', dpi = 300,bbox_inches='tight')



##Figure 5: County-level waterfall ACI analysis
fig, axs = plt.subplots(17,1,figsize=(30,60), gridspec_kw={'height_ratios': [8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]})
formatting = "{:,.1f}"
from matplotlib.ticker import FuncFormatter
index = np.array(ACI_list)[:]
data = median_rcp85_2041_2060
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
plt.plot(step.index, step.values, 'k', linewidth = 2)
plt.yticks(fontsize = 35)
plt.ylim(-120,10)
plt.title('RCP8.5 2040-2059', fontsize = 35, y =1.05)
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
plt.axhline(0, color='black', linewidth = 0.6, linestyle="dashed")
for i in (0,2,4,6,8,10,12):
    rect=mpatches.Rectangle([-0.5+i,-120], 1, 130, ec='white', fc='grey', alpha=0.2, clip_on=False)
    axs[0].add_patch(rect)
plt.tick_params(axis = 'x' , which = 'both', bottom = False, top = False, labelbottom = False)
#for i in (0,2,4,6,8,10):
 #   rect=mpatches.Rectangle([-0.5+i,-120], 1, 150, ec='white', fc='grey', alpha=0.2, clip_on=False)
  #  ax.add_patch(rect)
#plt.annotate('', xy = (0.055, -0.4), xycoords = 'axes fraction', xytext = (0.24,-0.4), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
#plt.annotate('Dormancy', xy = (0.095, -0.5), xycoords = 'axes fraction', fontsize = 35)
#plt.annotate('', xy = (0.295, -0.4), xycoords = 'axes fraction', xytext = (0.705,-0.4), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
#plt.annotate('Bloom (Pollination)', xy = (0.39,-0.5), xycoords = 'axes fraction', fontsize = 35)
#plt.annotate('', xy = (0.75, -0.4), xycoords = 'axes fraction', xytext = (0.87,-0.4), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
#plt.annotate('Growth', xy = (0.77, -0.5), xycoords = 'axes fraction', fontsize = 35)
#plt.annotate('', xy = (0.905, -0.4), xycoords = 'axes fraction', xytext = (0.965,-0.4), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
#plt.annotate('Harvest', xy = (0.89, -0.5), xycoords = 'axes fraction', fontsize = 35)
county_order_N_S = ['Tehama', 'Butte', 'Glenn', 'Yuba', 'Colusa', 'Sutter', 'Yolo', 'Solano', 'San Joaquin', 'Stanislaus', 'Madera', 'Merced', 'Fresno', 'Tulare', 'Kings', 'Kern']
county_order_N_S_num = [12, 0, 3, 15, 1, 11, 14, 9, 8, 10, 6, 7, 2,13, 5, 4]
for county in range(0,16):     
    formatting = "{:,.1f}"
    from matplotlib.ticker import FuncFormatter
    index = np.array(ACI_list)[:]
    data = aci_contribution_rcp85_county_2050_change_percent_median[county_order_N_S_num[county]]
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
    plt.text(s = str(county_order_N_S[county]), x = -3.35, y = -120, fontsize = 35)
    plt.plot(step.index, step.values, 'k', linewidth = 2)
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
    plt.ylim(-140,10)
    plt.axhline(0, color='black', linewidth = 0.6, linestyle="dashed")
    if county == 15:
        plt.xticks(np.arange(0,len(trans)), trans.index, rotation=90, fontsize = 35)
    else:
        plt.tick_params(axis = 'x' , which = 'both', bottom = False, top = False, labelbottom = False)
    for j in (0,2,4,6,8,10,12):
        rect=mpatches.Rectangle([-0.5+j,-140], 1, 150, ec='white', fc='grey', alpha=0.2, clip_on=False)
        axs[county+1].add_patch(rect)
plt.annotate('', xy = (0.055, -3.5), xycoords = 'axes fraction', xytext = (0.24,-3.5), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Dormancy', xy = (0.095, -4.3), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.295, -3.5), xycoords = 'axes fraction', xytext = (0.705,-3.5), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Bloom', xy = (0.46,-4.3), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.75, -3.5), xycoords = 'axes fraction', xytext = (0.87,-3.5), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Growth', xy = (0.77, -4.3), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.905, -3.5), xycoords = 'axes fraction', xytext = (0.965,-3.5), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Harvest', xy = (0.89, -4.3), xycoords = 'axes fraction', fontsize = 35)
plt.savefig(save_path+'waterfall_total_2050_rcp85.pdf', dpi = 300,bbox_inches='tight')



