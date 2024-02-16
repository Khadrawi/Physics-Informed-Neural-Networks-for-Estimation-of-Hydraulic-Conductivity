# from audioop import avg
# from distutils.log import error
# from turtle import distance
from turtle import position
import numpy as np
# import pandas as pdy7
# import math
import matplotlib.pyplot as plt
#import torch.nn.functional as F
# from datetime import datetime, timedelta
# from matplotlib.dates import datestr2num
# from tqdm import tqdm
# from tqdm import trange
# from models import *
from itertools import combinations
from scipy import interpolate

def K_func(theta, theta_r, theta_s, K_s, l, n):
    # theta vector, rest are constants
    m = 1-1/n
    S_e = (theta-theta_r)/(theta_s-theta_r)
    K = K_s*S_e**l * (1-(1-S_e**(1/m))**m)**2
    return K

fig_size = (10,8)
font = {
        'font.weight' : 'normal',
        'font.size'   : 20}
plt.rcParams.update(font)

hydrus_data = 'mixed'
moist_1_sorted_unique_array = np.load('./Results/'+hydrus_data+'/moist_1_sorted_unique_array.npy', allow_pickle=True)
median_K = np.load('./Results/'+hydrus_data+'/median_K.npy', allow_pickle=True)
# Remember to get this manually from PINN results
moist_and_K_gt = np.load('./Results/'+hydrus_data+'/moist_and_K_gt.npy', allow_pickle=True)

sensor_depth_list = [-1, -3, -5, -7, -9, -11, -13, -15, -17, -19] #in cm
sensor_combinations = list(combinations(sensor_depth_list, 3))

#  sensors with inner distance > 3 or 4 
valid_idx = []
for j in range(len(sensor_combinations)):
    if (sensor_combinations[j][0]-sensor_combinations[j][1]) <=3 and (sensor_combinations[j][1]-sensor_combinations[j][2]) <=3:
        valid_idx.append(j)


ref_moist = np.linspace(np.min(moist_and_K_gt[:,0]), 0.4, 500) #np.max(moist_and_K_gt[:,0])
K_sandy_loam = K_func(ref_moist, 0.065, 0.41, 106.1, 0.5, 1.89)
K_loam = K_func(ref_moist, 0.078, 0.43, 24.96, 0.5, 1.56)
K_silt_loam = K_func(ref_moist, 0.067, 0.45, 10.8, 0.5, 1.41)

K_sandy_loam[np.isnan(K_sandy_loam)] = 0
K_loam[np.isnan(K_loam)] = 0
K_silt_loam[np.isnan(K_silt_loam)] = 0

K_equiv = 100/(7/K_sandy_loam + 7/K_loam + 7/K_silt_loam)
K_ref = K_equiv/24 #converted to cm/hr

median_median_K = np.zeros([ ref_moist.size])
first_quart_median_K = np.zeros([ ref_moist.size])
third_quart_median_K = np.zeros([ ref_moist.size])
K_vals = np.zeros([len(sensor_combinations), len(ref_moist)])
errors = np.zeros([len(sensor_combinations), 1])
for j in range(len(sensor_combinations)):
    f = interpolate.interp1d(moist_1_sorted_unique_array[j], median_K[j], fill_value='extrapolate')
    K_vals[j,:] = f(ref_moist)
    errors[j] = np.sum((K_vals[j,:]-K_ref)**2)/np.sum(K_ref**2)

median_median_K = np.quantile(K_vals, 0.5, axis = 0)
first_quart_median_K = np.quantile(K_vals, 0.25, axis = 0)
third_quart_median_K = np.quantile(K_vals, 0.75, axis = 0)

# Best performance
ix = np.argmin(errors)
print(f'Best config: {sensor_combinations[ix]}, Error = {errors[ix].item():.5f}')
median_error = np.sum((median_median_K-K_ref)**2)/np.sum(K_ref**2)
print(f'\nMedian Error = {median_error.item():.5f}')
# second Best performance
ix2 = np.argsort(errors, axis = None)[1]

# ---- Plot by Averge distance between sensors ----
distance_dict = {}
for j in range(len(sensor_combinations)):
    avg_dist = (sensor_combinations[j][0]-sensor_combinations[j][1] + sensor_combinations[j][1]-sensor_combinations[j][2]) / 2
    if avg_dist in distance_dict:
        distance_dict[avg_dist].append(errors[j].item())
    else:
        distance_dict[avg_dist] = [errors[j].item()]
distance_dict = {key: np.mean(distance_dict[key]) for key in distance_dict.keys()}
distances =  np.array(list(distance_dict.keys()))
distances_sort_ix = np.argsort(distances)
distances = distances[distances_sort_ix]
distances_errors = np.array(list(distance_dict.values()))[distances_sort_ix]

# Equidistant sensors
equidistant_dict = {}
for j in range(len(sensor_combinations)):
    if (sensor_combinations[j][0]-sensor_combinations[j][1]) == (sensor_combinations[j][1]-sensor_combinations[j][2]):
        avg_dist = sensor_combinations[j][0]-sensor_combinations[j][1]
        if avg_dist in equidistant_dict:
            equidistant_dict[avg_dist].append(errors[j].item())
        else:
            equidistant_dict[avg_dist] = [errors[j].item()]

equidistant_dict = {key: np.mean(equidistant_dict[key]) for key in equidistant_dict.keys()}
equidistances =  np.array(list(equidistant_dict.keys()))
distances_sort_ix = np.argsort(equidistances)
equidistances = equidistances[distances_sort_ix]
equidistances_errors = np.array(list(equidistant_dict.values()))[distances_sort_ix]

# Average depth of the sensors
avg_depth_dict = {}
for j in range(len(sensor_combinations)):
    avg_depth = -(sensor_combinations[j][0] + sensor_combinations[j][1] + sensor_combinations[j][2]) / 3
    if avg_depth in avg_depth_dict:
        avg_depth_dict[avg_depth].append(errors[j].item())
    else:
        avg_depth_dict[avg_depth] = [errors[j].item()]

avg_depth_dict = {key: np.mean(avg_depth_dict[key]) for key in avg_depth_dict.keys()}
avg_depth =  np.array(list(avg_depth_dict.keys()))
distances_sort_ix = np.argsort(avg_depth)
avg_depth = avg_depth[distances_sort_ix]
avg_depth_errors = np.array(list(avg_depth_dict.values()))[distances_sort_ix]

# By depth of sensor "1" [0,1,2]
s1_depth_dict = {}
for j in range(len(sensor_combinations)):
    s1_depth_tmp = - sensor_combinations[j][1] 
    if s1_depth_tmp in s1_depth_dict:
        s1_depth_dict[s1_depth_tmp].append(errors[j].item())
    else:
        s1_depth_dict[s1_depth_tmp] = [errors[j].item()]

s1_depth_dict = {key: np.mean(s1_depth_dict[key]) for key in s1_depth_dict.keys()}
s1_depth =  np.array(list(s1_depth_dict.keys()))
distances_sort_ix = np.argsort(s1_depth)
s1_depth = s1_depth[distances_sort_ix]
s1_depth_errors = np.array(list(s1_depth_dict.values()))[distances_sort_ix]


# plt.figure(figsize=fig_size)
# plt.plot(errors, linewidth=3)
# plt.title('Errors for all 120 sensor combinations')


plt.figure(figsize=fig_size)
plt.plot(distances, distances_errors, linewidth=3)
plt.xlabel('Distance [cm]')
plt.ylabel(r'$\epsilon^k$')
plt.title('Errors by avg distance between sensors')
# plt.savefig('./Results/'+hydrus_data+'/Errors by avg distance between sensors.png', dpi=600)

plt.figure(figsize=fig_size)
plt.plot(equidistances, equidistances_errors, linewidth=3)
plt.xlabel('Distance [cm]')
plt.ylabel(r'$\epsilon^k$')
plt.title('Errors of equidistant sensors')
# plt.savefig('./Results/'+hydrus_data+'/Errors of equidistant sensors.png', dpi=600)

plt.figure(figsize=fig_size)
plt.plot(avg_depth, avg_depth_errors, linewidth=3)
plt.xlabel('Distance [cm]')
plt.ylabel(r'$\epsilon^k$')
plt.title('Errors by averge depth')
# plt.savefig('./Results/'+hydrus_data+'/Errors by averge depth.png', dpi=600)

plt.figure(figsize=fig_size)
plt.plot(s1_depth, s1_depth_errors, linewidth=3)
plt.xlabel('Distance [cm]')
plt.ylabel(r'$\epsilon^k$')
plt.title('Errors by the middle sensor depth "sensor 1"')
# plt.savefig('./Results/'+hydrus_data+'/Errors by the middle sensor depth.png', dpi=600)

plt.figure(figsize=fig_size)
plt.plot(ref_moist, median_median_K, color = 'tab:blue', linewidth=3, label='Median')
plt.fill_between(ref_moist, first_quart_median_K, third_quart_median_K, color = [(0.30,0.66,1)], label='1st to 3rd quartile')
plt.plot(ref_moist, K_ref, 'r', label='Ground Truth')
plt.xlabel(r'$\theta \quad [cm^3/cm^3]$')
plt.ylabel(r'$K(\theta)\quad [cm/hr]$')
plt.legend(loc='upper left')
plt.title(r"$K(\theta)$")
# plt.savefig('./Results/'+hydrus_data+'/Errors 120 combinations.png', dpi=600)

# # Distance >=3

# median_median_K_valid = np.quantile(K_vals[valid_idx], 0.5, axis = 0)
# first_quart_median_K_valid = np.quantile(K_vals[valid_idx], 0.25, axis = 0)
# third_quart_median_K_valid = np.quantile(K_vals[valid_idx], 0.75, axis = 0)

# plt.figure(figsize=fig_size)
# plt.plot(ref_moist, median_median_K_valid, color = 'tab:blue', linewidth=3, label='Median')
# plt.fill_between(ref_moist, first_quart_median_K_valid, third_quart_median_K_valid, color = [(0.30,0.66,1)], label='1st to 3rd quartile')
# plt.plot(ref_moist, K_ref, 'r', label='Ground Truth')
# plt.xlabel(r'$\theta \quad [cm^3/cm^3]$ - Sensor 1')
# plt.ylabel(r'$K(\theta)\quad [cm/hr]$')
# plt.legend(loc='upper left')
# plt.title(r"$K(\theta)$ - Distance >=3")

plt.figure(figsize=fig_size)
plt.plot(ref_moist, K_vals[ix,:], color = 'tab:blue', linewidth=3, label='Median')
plt.plot(ref_moist, K_ref, 'r', label='Ground Truth')
plt.ylabel(r'$K(\theta)\quad [cm/hr]$')
plt.xlabel(r'$\theta \quad [cm^3/cm^3]$')
plt.legend(loc='upper left')
plt.title(r"Best $K(\theta)$")
# plt.savefig('./Results/'+hydrus_data+'/Best config.png', dpi=600)

# plt.figure()
# plt.plot(ref_moist, K_vals[ix2,:], color = 'tab:blue', linewidth=3, label='Median')
# plt.plot(ref_moist, K_ref, 'r', label='Ground Truth')
# plt.ylabel(r'$K(\theta)\quad [cm/hr]$')
# plt.legend()
# plt.title(r"Second Best $K(\theta)$")
plt.show()

plt.pause(0.1)