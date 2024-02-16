# -*- coding: utf-8 -*-
from numpy.core.fromnumeric import mean, size
from numpy.core.numeric import NaN
import torch
import numpy as np
import math as math
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange
from preprocess import csvdata, get_features
from models_and_training import *
from itertools import combinations
from scipy import interpolate
use_cuda = torch.cuda.is_available()
device = torch.device("cpu")
torch.manual_seed(0)
plt.rcParams.update({'font.size': 14})

# device = torch.device("cuda:0" if use_cuda else "cpu")

# ----Network and data params------
all_sensors = {'moist_0':0, 'moist_1':1, 'moist_2':2}
used_input = [True, True, True] # Choose which out of "all_sensors" will be used: moist[:,[0,1,2]]
# Hyper Parameters
loaded_num_past_data = 15 # Number of past data points inculded in data set for each sensor
used_num_past_data = 2 #2 #20 # Number of past data used as network inputs. NOTE must be <= AutoReg_index
avg_window_length = 1 #8
LR = 0.002
EPOCHS = 500 # 500
BATCH_SIZE = 80 #80
train_reps = 1 #10 #100

dataset = 'Hydrus'
date_column = 12 
depth_column = (2) 
moist_columns  = (4) 
K_column  = (5) 
hydrus_data = 'sandy_loam' # 'sandy_loam', 'loam', 'silt_loam' 'mixed'
file_name = '../Hydrus data/' + hydrus_data
time_delta = 0.012*24*3600 # time_step in hydrus = 0.012 days, converting it to seconds

used_meas = {key:i for i, key in enumerate(all_sensors) if used_input[i]}
available_meas_with_idx = {'_'.join([meas,str(i)]):i*len(used_meas)+j for i in range(loaded_num_past_data+1) for j,meas in enumerate(used_meas)}
# description of indexes in available_meas_... : sensor_z_t, z: sensor location[0,1,2], t: past time index[0,1,2...]
# available_meas['cd']=len(available_meas)
used_input = used_input * (loaded_num_past_data+1)

# load data
dataobj = csvdata(file_name, time_delta, loaded_num_past_data)

# Sensor depth to be considered for training
sensor_depth_list = [-1, -3, -5, -7, -9, -11, -13, -15, -17, -19] #in cm
# Use all combinations of 3 sensors within the list
sensor_combinations = list(combinations(sensor_depth_list, 3))
dataobj.read(loaded=False)

moist_1_sorted_unique_array = []
median_D = []
median_dD_dtheta = []
median_K = []
for i, sensor_depth in enumerate(sensor_combinations):
    print('\nUsing depths :', sensor_depth,'  ',i,'/',len(sensor_combinations))
    trainset_raw = dataobj.get_data(sensor_depth)
    dmoist_dt_train, dmoist_dz_train, dmoist_dz_square_train, lapl_train, theta_1_train = get_features(trainset_raw, available_meas_with_idx, used_num_past_data, delta_z = -1*np.diff(sensor_depth), delta_t = time_delta/60**2, moving_average_window_length=avg_window_length)
    # dmoist_dt_test, dmoist_dz_test, dmoist_dz_square_test, lapl_test, theta_1_test= get_features(testset_raw, available_meas_with_idx, used_num_past_data, delta_z = -1*np.diff(sensor_depth), delta_t = time_delta/60**2, window_length=avg_window_length, theta_1_past_idx = theta_1_past_idx)

    ####--- get_feature_map function, that shows the columns(indices) corresponding to each input feature
    b = [0, dmoist_dz_train.shape[1], dmoist_dz_square_train.shape[1], lapl_train.shape[1], theta_1_train.shape[1]]
    c = np.cumsum(b)
    feature_names = ["dmoist_dz", "dmoist_dz_square", "lapl", "theta_1"]

    feature_map = {key:torch.arange(c[i], c[i+1]).to(device) for i, key in enumerate(feature_names)}
    ####---end function

    train_features = np.hstack((dmoist_dz_train, dmoist_dz_square_train, lapl_train, theta_1_train))
    # from numpy to torch
    trainset_t, dmoist_dt_train_t  = torch.Tensor(train_features).to(device), torch.Tensor(dmoist_dt_train).to(device)

    torch_train_dataset = torch.utils.data.TensorDataset(trainset_t, dmoist_dt_train_t)# X: first and second derivative wrt to space, Y: first derivative wrt time

    # !!!!NOTE!!!!!include or not theta 1 in net input
    keep_theta_1= True
    if keep_theta_1:
        input_size = train_features.shape[1] 
    else:
        input_size = train_features.shape[1] - len(feature_map['theta_1'])

    train_set_dataloader = torch.utils.data.DataLoader(
        dataset = torch_train_dataset, # [indices[mask].flatten()],  # torch TensorDataset format
        batch_size = BATCH_SIZE,  # mini batch size
        shuffle = True,
    )

    # Sensor 1 (@middle) moisture data (t=0 current time)
    moist_1_train = theta_1_train[:,0] # same as     moist_1_train = trainset_raw[:, available_meas_with_idx['moist_1_0']]

    # Dmoist_dz @Sensor1 central difference
    dmoist_dz_1_train = 0.5*(dmoist_dz_train[:, 0] + dmoist_dz_train[:, used_num_past_data+1])
    delta_z = -1*np.diff(sensor_depth)
    Model_min, D_all, dD_dtheta_all, D_integrated_all, K_all, moist_1_sorted_unique = train_models_loop(\
        train_reps, train_set_dataloader, trainset_t, dmoist_dt_train_t, delta_z, EPOCHS, moist_1_train, dmoist_dz_1_train, \
        feature_map, input_size, used_num_past_data, LR, device, keep_theta_1=keep_theta_1 )

    # Plot Diffusivity (direct not integrated)
    moist_1_sorted_unique_array.append(moist_1_sorted_unique)
    median_D.append(np.quantile(D_all, 0.5, axis = 0))
    median_dD_dtheta.append(np.quantile(dD_dtheta_all, 0.5, axis = 0))
    median_K.append(np.quantile(K_all, 0.5, axis = 0))


np.save('./Results/'+hydrus_data+'/moist_1_sorted_unique_array.npy', moist_1_sorted_unique_array)
np.save('./Results/'+hydrus_data+'/median_D.npy', median_D)
np.save('./Results/'+hydrus_data+'/median_dD_dtheta.npy', median_dD_dtheta)
np.save('./Results/'+hydrus_data+'/median_K.npy', median_K)


ref_moist = np.linspace(np.min(dataobj.moist_and_K[:,0]), np.max(dataobj.moist_and_K[:,0]), 1400)
median_median_D = np.zeros([ ref_moist.size])
first_quart_median_D = np.zeros([ ref_moist.size])
third_quart_median_D = np.zeros([ ref_moist.size])
median_median_K = np.zeros([ ref_moist.size])
first_quart_median_K = np.zeros([ ref_moist.size])
third_quart_median_K = np.zeros([ ref_moist.size])
median_median_dD_dtheta = np.zeros([ ref_moist.size])
first_quart_median_dD_dtheta = np.zeros([ ref_moist.size])
third_quart_median_dD_dtheta = np.zeros([ ref_moist.size])

f = interpolate.interp1d(dataobj.moist_and_K[:,0], dataobj.moist_and_K[:,1])
K_ref = f(ref_moist)
D_vals = np.zeros([len(sensor_combinations), len(ref_moist)])
K_vals = np.zeros([len(sensor_combinations), len(ref_moist)])
dD_vals = np.zeros([len(sensor_combinations), len(ref_moist)])
for j in range(len(sensor_combinations)):
     f = interpolate.interp1d(moist_1_sorted_unique_array[j], median_D[j], fill_value='extrapolate')
     D_vals[j,:] = f(ref_moist)
     f = interpolate.interp1d(moist_1_sorted_unique_array[j], median_K[j], fill_value='extrapolate')
     K_vals[j,:] = f(ref_moist)
     f = interpolate.interp1d(moist_1_sorted_unique_array[j], median_dD_dtheta[j], fill_value='extrapolate')
     dD_vals[j,:] = f(ref_moist)
median_median_D = np.quantile(D_vals, 0.5, axis = 0)
first_quart_median_D = np.quantile(D_vals, 0.25, axis = 0)
third_quart_median_D = np.quantile(D_vals, 0.75, axis = 0)
median_median_dD_dtheta = np.quantile(dD_vals, 0.5, axis = 0)
first_quart_median_dD_dtheta = np.quantile(dD_vals, 0.25, axis = 0)
third_quart_median_dD_dtheta = np.quantile(dD_vals, 0.75, axis = 0)
median_median_K = np.quantile(K_vals, 0.5, axis = 0)
first_quart_median_K = np.quantile(K_vals, 0.25, axis = 0)
third_quart_median_K = np.quantile(K_vals, 0.75, axis = 0)


fig_size = (11,7)

plt.figure(figsize=fig_size)
plt.plot(ref_moist, median_median_D, color = 'tab:blue', linewidth=2, label='Median')
plt.fill_between(ref_moist, first_quart_median_D, third_quart_median_D, color = [(0.30,0.66,1)], label='1st to 3rd quartile')
plt.xlabel(''.join([r'$ \theta \quad [cm^3/cm^3]$ - Sensor 1']))
plt.ylabel(r'$D(\theta) \quad [cm^2/hr]$')
plt.legend()
plt.title(r"$ D ( \theta )$ - Sensor 1""\n Fit on training data")
plt.pause(0.1)
# plt.savefig('Figs/D.png', dpi=1500)

# Plot dD/dtheta

plt.figure(figsize=fig_size)
plt.plot(ref_moist, median_median_dD_dtheta, color = 'tab:blue', linewidth=2, label='Median')
plt.fill_between(ref_moist, first_quart_median_dD_dtheta, third_quart_median_dD_dtheta, color = [(0.30,0.66,1)], label='1st to 3rd quartile')
plt.xlabel(''.join([r'$ \theta \quad [cm^3/cm^3]$ - Sensor 1']))
plt.ylabel(r'$\frac{dD(\theta)}{d\theta} \quad [cm^2/hr]$')
plt.legend()
plt.title(r"$\frac{dD(\theta)}{d\theta}$ - Mean values of 100 trials - Sensor 1")
plt.pause(0.1)
# plt.savefig('Figs/dD_dtheta.png', dpi=1500)


plt.figure(figsize=fig_size)
plt.plot(ref_moist, median_median_K, color = 'tab:blue', linewidth=2, label='Median')
plt.fill_between(ref_moist, first_quart_median_K, third_quart_median_K, color = [(0.30,0.66,1)], label='1st to 3rd quartile')
plt.plot(dataobj.moist_and_K[:,0], dataobj.moist_and_K[:,1], 'r', label='Ground Truth')
plt.xlabel(''.join([r'$\theta \quad [cm^3/cm^3]$ - Sensor 1']))
plt.ylabel(r'$K(\theta)\quad [cm/hr]$')
plt.legend()
plt.title(r"$K(\theta)$ - integrated from mean of $dK/d\theta$")
plt.pause(0.1)
# plt.savefig('Figs/K.png', dpi=1500)

plt.show()
# input("<Hit Enter To Close>")
# plt.close()

print('')
