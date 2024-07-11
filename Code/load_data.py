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

def load_data(hydrus, architec):
    # hydrus = 'loam'
    # architec = ['PINN','Finite_diff']
    path =  architec+'/' + hydrus
    if architec == 'Finite_diff':
        moist_1_sorted_unique_array = np.load(path+'/moist_1_sorted_unique_array.npy', allow_pickle=True)
        median_K = np.load(path +'/median_K.npy', allow_pickle=True)
    else:
        moist_and_K = np.load(path + '/moist_and_K.npy', allow_pickle=True)

    # Remember to get this manually from PINN results
    moist_and_K_gt = np.load(path + '/moist_and_K_gt.npy', allow_pickle=True)

    sensor_depth_list = [-1, -3, -5, -7, -9, -11, -13, -15, -17, -19] #in cm
    sensor_combinations = list(combinations(sensor_depth_list, 3))

    if hydrus == 'mixed':
        ref_moist = np.linspace(np.min(moist_and_K_gt[:,0]), 0.4, 500)
        K_sandy_loam = K_func(ref_moist, 0.065, 0.41, 106.1, 0.5, 1.89)
        K_loam = K_func(ref_moist, 0.078, 0.43, 24.96, 0.5, 1.56)
        K_silt_loam = K_func(ref_moist, 0.067, 0.45, 10.8, 0.5, 1.41)

        K_sandy_loam[np.isnan(K_sandy_loam)] = 0
        K_loam[np.isnan(K_loam)] = 0
        K_silt_loam[np.isnan(K_silt_loam)] = 0

        K_equiv = 100/(7/K_sandy_loam + 7/K_loam + 7/K_silt_loam)
        ref_K = K_equiv/24 #converted to cm/hr

    else: 
        ref_moist = np.linspace(np.min(moist_and_K_gt[:,0]), np.max(moist_and_K_gt[:,0]), 500)
        f = interpolate.interp1d(moist_and_K_gt[:,0], moist_and_K_gt[:,1])
        ref_K = f(ref_moist)/24 #converted to cm/hr

    # median_median_K = np.zeros([ ref_moist.size])
    # first_quart_median_K = np.zeros([ ref_moist.size])
    # third_quart_median_K = np.zeros([ ref_moist.size])
    K_vals = np.zeros([len(sensor_combinations), len(ref_moist)])
    errors = np.zeros([len(sensor_combinations), 1])

    for j in range(len(sensor_combinations)):
        if architec == 'Finite_diff':
            f = interpolate.interp1d(moist_1_sorted_unique_array[j], median_K[j], fill_value='extrapolate')
            K_vals[j,:] = f(ref_moist)
        else:
            f = interpolate.interp1d(moist_and_K[j][:,0], moist_and_K[j][:,1], fill_value='extrapolate')
            K_vals[j,:] = f(ref_moist)/24 #note the unit conversion

        errors[j] = np.sum((K_vals[j,:]-ref_K)**2)/np.sum(ref_K**2)

    median_K = np.quantile(K_vals, 0.5, axis = 0)
    first_quart_K = np.quantile(K_vals, 0.25, axis = 0)
    third_quart_K = np.quantile(K_vals, 0.75, axis = 0)

    # Best performance
    ix = np.argmin(errors)
    median_error = np.sum((median_K-ref_K)**2)/np.sum(ref_K**2)
    print(f'\nMedian Error {architec} {hydrus} = {median_error.item():.5f}')
    print(f'Best config: {sensor_combinations[ix]}, Error = {errors[ix].item():.5f}')

    # ---- Plot by Averge distance between sensors ----
    distance_dict = {}
    for j in range(len(sensor_combinations)):
        avg_dist = (sensor_combinations[j][0]-sensor_combinations[j][1] + sensor_combinations[j][1]-sensor_combinations[j][2]) / 2
        if avg_dist in distance_dict:
            distance_dict[avg_dist].append(errors[j].item())
        else:
            distance_dict[avg_dist] = [errors[j].item()]
    distance_dict = {key: np.mean(distance_dict[key]) for key in distance_dict.keys()}
    avg_distances =  np.array(list(distance_dict.keys()))
    distances_sort_ix = np.argsort(avg_distances)
    avg_distances = avg_distances[distances_sort_ix]
    avg_distances_errors = np.array(list(distance_dict.values()))[distances_sort_ix]

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
    
    out_dict = {'ref_moist': ref_moist, 'ref_K':ref_K, 'median_K':median_K, 'first_quart_K':first_quart_K, 'third_quart_K':third_quart_K, 
                'avg_distances':avg_distances, 'avg_distances_errors':avg_distances_errors, 'equidistances':equidistances,
                'equidistances_errors':equidistances_errors, 'avg_depth':avg_depth, 'avg_depth_errors':avg_depth_errors,
                's1_depth':s1_depth, 's1_depth_errors':s1_depth_errors, 'best_K': K_vals[ix,:]}
    return out_dict

def load_different_soils(hydrus_list, architec):
    dict_out = {}
    for j in hydrus_list:
        out = load_data(j, architec)
        dict_out[j] = out
    return dict_out