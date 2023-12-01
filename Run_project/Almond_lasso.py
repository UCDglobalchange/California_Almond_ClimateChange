##Code to run LASSO regression model with gridMET-ACI (X) and historical almond yield (Y)
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import netCDF4 as nc
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
from sklearn import preprocessing
from numpy import genfromtxt
import seaborn as sns
from numpy import savetxt
from sklearn.linear_model import LassoCV
import yellowbrick
from yellowbrick.regressor import cooks_distance
import sys

home_path='/home/shqwu/California_Almond_ClimateChange-main/Run_project'
save_path = home_path+'/intermediate_data/lasso_model/'
input_path_ACI = home_path+'/intermediate_data/Gridmet_csv/'
input_path_yield = home_path+'/input_data/'

trial=np.int64(sys.argv[1])
aci_num=14
model_list = ['bcc-csm1-1','bcc-csm1', 'BNU-ESM', 'CanESM2', 'CSIRO-Mk3-6-0', 'GFDL-ESM2G', 'GFDL-ESM2M', 'inmcm4', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR','CNRM-CM5', 'HadGEM2-CC365','HadGEM2-ES365', 'IPSL-CM5B-LR', 'MIROC5', 'MIROC-ESM', 'MIROC-ESM-CHEM']
X = genfromtxt(input_path_ACI+'Gridmet.csv', delimiter = ',')
Y = genfromtxt(input_path_yield+'almond_yield_1980_2020.csv', delimiter = ',')[:,1:].flatten('F')
distance = yellowbrick.regressor.influence.cooks_distance(X,Y, show = False).distance_ ##data remove by cook's distance
threhold = yellowbrick.regressor.influence.cooks_distance(X,Y, show = False).influence_threshold_
X=np.delete(X, np.where(distance>threhold), axis = 0)
Y=np.delete(Y, np.where(distance>threhold), axis = 0)

coef = np.zeros((aci_num*2+32))
score_sum = np.zeros((1))
score_test = np.zeros((1))
score_train = np.zeros((1))


times = 0
while times <=0:
    X_train,X_test,y_train,y_test=train_test_split(X,Y, test_size=0.3)
    lassocv = LassoCV(alphas=[0,5,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001,0.000005,0.000001,0.0000005], max_iter = np.int64(10e6), fit_intercept=False, random_state = trial)
    lassocv.fit(X_train, y_train) 
    coef[:] = lassocv.coef_
    score_sum[:] = lassocv.score(X,Y)
    score_test[:] = lassocv.score(X_test,y_test)
    score_train[:] = lassocv.score(X_train,y_train)
    times = times+1
savetxt(str(save_path)+'coef_'+str(trial)+'.csv', coef, delimiter = ',')
savetxt(str(save_path)+'score_'+str(trial)+'.csv', score_sum, delimiter = ',')
savetxt(str(save_path)+'score_train_'+str(trial)+'.csv', score_train, delimiter = ',')
savetxt(str(save_path)+'score_test_'+str(trial)+'.csv', score_test, delimiter = ',')


