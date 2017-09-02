""""
Below are suporting functions
"""
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge,BayesianRidge
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error
import numpy as np

# outlier clean
def remove_outlier(data,duration_control,passenger_count_control):
	# trip duration clean up
	trip_mean, trip_std = np.mean(data['trip_duration']), np.std(data['trip_duration'])
	data = data[data['trip_duration']<=trip_mean + duration_control*trip_std]
	data = data[data['trip_duration']>=trip_mean - duration_control*trip_std]

	# geographic clean up
	data = data[data['pickup_longitude'] <= -73.75]
	data = data[data['pickup_longitude'] >= -74.03]
	data = data[data['pickup_latitude'] <= 40.85]
	data = data[data['pickup_latitude'] >= 40.63]
	data = data[data['dropoff_longitude'] <= -73.75]
	data = data[data['dropoff_longitude'] >= -74.03]
	data = data[data['dropoff_latitude'] <= 40.85]
	data = data[data['dropoff_latitude'] >= 40.63]

	# passenger count
	data = data[data['passenger_count'] <= passenger_count_control]

	return data

# time massage
def time_massage(data):
	data['pickup_datetime'] = pd.to_datetime(data.pickup_datetime)
	data.loc[:, 'pickup_month'] = data['pickup_datetime'].dt.month
	data.loc[:, 'pickup_hour'] = data['pickup_datetime'].dt.hour
	data.loc[:,'pickup_half_hour'] = data['pickup_hour']*2+(data['pickup_datetime'].dt.minute/30).apply(np.floor)
	data.loc[:, 'pickup_day_of_week'] = data['pickup_datetime'].dt.dayofweek
	
	return data

def time_categorize(data,hour_control):
	if hour_control == 'full':
		cols = ['pickup_month','pickup_hour','pickup_day_of_week']
	elif hour_control == 'half':
		cols = ['pickup_month','pickup_half_hour','pickup_day_of_week']
	
	for i in cols:
		data[i] = data[i].astype(str)
    
	data = data.join(pd.get_dummies(
    	data[cols],
    	prefix=cols,prefix_sep='_')
    )

	return data

# vendor categorize
def vendor_categorize(data):
	return data.join(pd.get_dummies(data['vendor_id'], prefix='vendor', prefix_sep='_'))

# passenger_count categorize
def passenger_count_categorize(data):
	return data.join(pd.get_dummies(data['passenger_count'], prefix='passenger_count', prefix_sep='_'))

# store_and_forward_flag categorize:
def store_and_fwd_flag_categorize(data):
	return data.join(pd.get_dummies(data['store_and_fwd_flag'], prefix='store_and_fwd_flag', prefix_sep='_'))

# create clusters based on two features (lon, lat)
# or three dimensions (time, lon, lat)
def cluster_generator(train,test,sample_percent,n_clusters,batch_size):
	coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values))
	# sample from data to train kmean
	sample_inds = np.random.permutation(len(coords))[:int(len(coords)*sample_percent)]
	kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size).fit(coords[sample_inds])
	# make predictions
	train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
	train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])
	test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
	test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])
	
	return {'train':train,'test':test}

def cluster_categorize(data):
	data = data.join(pd.get_dummies(data['pickup_cluster'], prefix='pickup_cluster', prefix_sep='_'))

	return data.join(pd.get_dummies(data['dropoff_cluster'], prefix='dropoff_cluster', prefix_sep='_'))

# get route information
# fastest time, second fastest time, # of left/right turns
def route_info_generator(data,fast_route):
	left_count = lambda x: x.split("|").count('left')
	right_count = lambda x: x.split("|").count('right')
	fast_route['left_freq'] = fast_route['step_direction'].apply(left_count)
	fast_route['right_freq'] = fast_route['step_direction'].apply(right_count)
	cols = ['id','total_distance','total_travel_time','left_freq','right_freq']
	fast_route_sub = fast_route[cols]
	
	return pd.merge(data,fast_route_sub,on='id',how='left')

# prepare data for model training
def data_prepare(data,data_type):
	if data_type == 'train':
		data['log_trip_duration'] = np.log(data['trip_duration'].values + 1)
		cols = ['id', 'vendor_id', 'pickup_datetime', 'dropoff_datetime', 'passenger_count', \
			'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', \
			'store_and_fwd_flag', 'trip_duration', 'pickup_month', \
			'pickup_hour', 'pickup_half_hour', 'pickup_day_of_week', 'pickup_cluster', 'dropoff_cluster']
	elif data_type == 'test':
		cols = ['id', 'vendor_id', 'pickup_datetime', 'passenger_count', \
			'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', \
			'store_and_fwd_flag', 'pickup_month', \
			'pickup_hour', 'pickup_half_hour', 'pickup_day_of_week', 'pickup_cluster', 'dropoff_cluster']
	return data.drop(cols, axis = 1)

# model train
def model_train(data,test):
	train, valid = train_test_split(data, test_size = 0.2)
	x_train = train.drop(['log_trip_duration'], axis=1)
	y_train = train['log_trip_duration']
	x_valid = valid.drop(['log_trip_duration'], axis=1)
	y_valid = valid['log_trip_duration']

	matrix_train = xgb.DMatrix(x_train, label=y_train)
	matrix_valid = xgb.DMatrix(x_valid, label=y_valid)
	matrix_test = xgb.DMatrix(test)
	watchlist = [(matrix_train, 'train'), (matrix_valid, 'valid')]

	xgb_pars = {'min_child_weight': 1, 'eta': 0.5, 'colsample_bytree': 0.9,
				'max_depth': 6, 'subsample': 0.9, 'lambda': 1., 'nthread': -1, 
				'booster' : 'gbtree', 'silent': 1,
				'eval_metric': 'rmse', 'objective': 'reg:linear'}
	model = xgb.train(xgb_pars, matrix_train, 10, watchlist,
			early_stopping_rounds=2,maximize=False, verbose_eval=1)
	
	print('Modeling RMSLE %.5f' % model.best_score)

	return model
