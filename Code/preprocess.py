# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import glob

class csvdata():

    def __init__(self, filename, time_delta, num_past_data):
        self.num_sensors = 3 # number of moisture data sensors in the study is 3
        self.num_past_data = num_past_data # number of past data points from each sensor, to add to the training data
        self.filename = filename
        self.time_column = 12
        self.depth_column = 2
        self.moist_column = 4
        self.K_column = 5
        self.data_ = []
        self.data_rain = None
        self.dates_training = None
        self.dates_testing = None
        self.moist_and_K = [] # will contain theta(moisture) and K from the first loaded csv file only. Used to align performance metrics for comparaison between models
        self.time_delta = time_delta # time step between each data point
 
    def dataset_with_pastpoints(self, moisture):
        trainset = np.empty((moisture.shape[0]-self.num_past_data, self.num_sensors*(self.num_past_data+1)))
        # This is basically  np.roll or PDdataframe.shift, but no fill value, lines are removed
        for i in range(self.num_past_data+1):
            trainset[:,i*self.num_sensors:(i+1)*self.num_sensors] = moisture[self.num_past_data-i:(-i if i!=0 else None),:]
        self.full_trainset = trainset
        return trainset

    def get_moisture(self, depths, file_index):
        time = np.unique(self.data_[file_index]['time'])
        tmp_moist_data = np.zeros((time.shape[0], len(depths)))
        for i in range(len(depths)):
            # idx = self.data_[file_index][:, 1] == depths[i]
            # next following 3 lines when the depths don't equal exactly the simulated depths, otherwise use line above
            existing_depths = np.unique(self.data_[file_index]['depth'])
            closest_depth = np.argmin(np.abs(existing_depths-depths[i]))
            idx = self.data_[file_index]['depth'] == existing_depths[closest_depth]
            tmp_moist_data[:,i] = self.data_[file_index]['theta'][idx].values

        # Saving moisture(theta) and K values from first file only
        if file_index==0:
            moist_and_K = self.data_[file_index][['theta', 'K']].values # adding K 
            _, idx = np.unique(moist_and_K[:,0], return_index=True)
            moist_and_K = moist_and_K[idx,:]
            # turn cm/days to cm/hours
            moist_and_K[:,1] = moist_and_K[:,1]/24
            self.moist_and_K = moist_and_K[np.argsort(moist_and_K[:,0]),:]
   
        return tmp_moist_data

    def read(self, loaded=False): 
        listdir_ = glob.glob(self.filename+'/*.csv')
        num_of_files = len(listdir_)
        if not loaded:
            for i, csv_file in enumerate(listdir_):
                if i < num_of_files:

                    self.data_.append(pd.read_csv(csv_file, delimiter=',',
                                         dtype=float))
                else:
                    break
            np.save('loaded_data.npy', self.data_)
        else:
            self.data_ = np.load('loaded_data.npy', allow_pickle=True)
            
    def get_data(self, depths):
        trainset_raw = np.zeros((0, self.num_past_data*3+3))
        for i in range(len(self.data_)):
            moisture = self.get_moisture(depths, i)
            trainset_raw_tmp = self.dataset_with_pastpoints(moisture)
            trainset_raw = np.vstack([trainset_raw, trainset_raw_tmp])
        return trainset_raw

def first_order_time_derivative(input, idx_map, delta_t, moving_average_window_length=3):
    
    # first get the derivatives of past window_length !!!! make sure window_length<available past points !!!
    derivatives = np.empty((input.shape[0],moving_average_window_length))
    for i in range(moving_average_window_length):
        derivatives[:,i] = (input[:,idx_map["moist_1_"+str(i)]] - input[:,idx_map["moist_1_"+str(i+1)]]) /(delta_t)
    # then apply filter
    out = np.mean(derivatives, axis = 1)
    return out

def get_features(input, idx_map, used_autoreg_idx, delta_z = [], delta_t = 1, moving_average_window_length=3, theta_1_past_idx=0):
    # NOTE: used_autoreg_idx is the effective number of past points we want to use as the network's inputs
    # Function to get first derivative of time also the first and second derivative of space
    # window_length: moving average window length
    # # First order time derivative with a specific moving average derivative window
    dmoist_dt = first_order_time_derivative(input, idx_map, delta_t, moving_average_window_length)
    lapl = np.zeros((len(input), used_autoreg_idx+1)) # second derivative from t0 to t_autoreg_idx wrt to space
    # dmoist_dz = np.zeros((len(input), 4*(used_autoreg_idx+1))) # first derivative and squared from t0 to t_autoreg_idx wrt space, using s2-s1 and s1-s0 seperately not the mean s2-s0/2
    dmoist_dz = np.zeros((len(input), 2*(used_autoreg_idx+1))) # first derivative from t0 to t_autoreg_idx wrt space, using s2-s1 and s1-s0 seperately not the mean s2-s0/2
    dmoist_dz_square = np.zeros((len(input), 2*(used_autoreg_idx+1))) # first derivative from t0 to t_autoreg_idx wrt space, using s2-s1 and s1-s0 seperately not the mean s2-s0/2
    theta_1 = np.zeros((len(input), theta_1_past_idx+1))
    for i in range(used_autoreg_idx+1):    
        # second derivative
        lapl[:,i] = ((input[:,idx_map["moist_2_"+str(i)]]-input[:,idx_map["moist_1_"+str(i)]]) /delta_z[1] - (input[:,idx_map["moist_1_"+str(i)]]-input[:,idx_map["moist_0_"+str(i)]])/delta_z[0] )/ (np.sum(delta_z)/2)
        #first derivatives
        dmoist_dz_1 = (input[:,idx_map["moist_2_"+str(i)]] - input[:,idx_map["moist_1_"+str(i)]] ) / (delta_z[1])
        dmoist_dz_2 = (input[:,idx_map["moist_1_"+str(i)]] - input[:,idx_map["moist_0_"+str(i)]] ) / (delta_z[0])
        # dmoist_dz[:,i*4:4*(i+1)] = np.hstack((dmoist_dz_1[:, np.newaxis], dmoist_dz_2[:, np.newaxis], dmoist_dz_1[:, np.newaxis]**2, dmoist_dz_2[:, np.newaxis]**2))
        dmoist_dz[:,i*2:2*(i+1)] = np.hstack((dmoist_dz_1[:, np.newaxis], dmoist_dz_2[:, np.newaxis]))
        dmoist_dz_square[:,i*2:2*(i+1)] = np.hstack(( dmoist_dz_1[:, np.newaxis]**2, dmoist_dz_2[:, np.newaxis]**2))
        if i<=theta_1_past_idx:
            theta_1[:,i] = input[:,idx_map["moist_1_"+str(i)]]
    
    return dmoist_dt[:, np.newaxis], dmoist_dz, dmoist_dz_square, lapl, theta_1