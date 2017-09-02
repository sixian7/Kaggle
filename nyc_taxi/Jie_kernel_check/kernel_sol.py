""""
project: New York taxi prediction
creator: Jie Yang
date: 08/31/2017
supporting source: https://www.kaggle.com/karelrv/nyct-from-a-to-z-with-xgboost-tutorial
"""
from utils import *
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge,BayesianRidge
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error
import numpy as np
 

# load data
train = pd.read_csv('/Users/jieyang/Documents/ml/kaggle/ny_taxi/train.csv')
test = pd.read_csv('/Users/jieyang/Documents/ml/kaggle/ny_taxi/test.csv')

fast_route_train_1 = pd.read_csv('/Users/jieyang/Documents/ml/kaggle/ny_taxi/fastest_routes_train_part_1.csv')
fast_route_train_2 = pd.read_csv('/Users/jieyang/Documents/ml/kaggle/ny_taxi/fastest_routes_train_part_2.csv')
fast_route_train = pd.concat([fast_route_train_1,fast_route_train_2])

fast_route_test = pd.read_csv('/Users/jieyang/Documents/ml/kaggle/ny_taxi/fastest_routes_test.csv')

# clean training data
train = remove_outlier(train,3,8)
# datatime convert for both train and test
train = time_massage(train)
test = time_massage(test)
# convert time data to categorical for both train and test
train = time_categorize(train,'full')
test = time_categorize(test,'full')
# vendor categorization
train = vendor_categorize(train)
test = vendor_categorize(test)
# passenger count categorization
train = passenger_count_categorize(train)
test = passenger_count_categorize(test)
# store flag
train = store_and_fwd_flag_categorize(train)
test = store_and_fwd_flag_categorize(test)
# clustering
clustering = cluster_generator(train,test,0.5,10,10000)
train,test = clustering['train'],clustering['test']
# cluster id categorization
train = cluster_categorize(train)
test = cluster_categorize(test)
# add route information
train = route_info_generator(train,fast_route_train)
test = route_info_generator(test,fast_route_test)
# prepare data
train = data_prepare(train,'train')
test = data_prepare(test,'test')

print (train.head(2))

model = model_train(train,test)










