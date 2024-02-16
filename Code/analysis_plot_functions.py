# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pdy7
import random
import math as math
import matplotlib.pyplot as plt
#import torch.nn.functional as F
from datetime import datetime, timedelta
from matplotlib.dates import datestr2num
from models_and_training import *
import scipy.integrate as integrate
use_cuda = torch.cuda.is_available()
device = torch.device("cpu")

def D_fun(D, moist_1_sorted, moist1_idx_sorted, moist_1_sorted_unique):
    D_sorted_unique_moist = np.zeros_like(moist_1_sorted_unique)
    D_sorted_moist = D[moist1_idx_sorted]
    for i in range(len(moist_1_sorted_unique)):
        idx = moist_1_sorted == moist_1_sorted_unique[i]
        D_sorted_unique_moist[i] = np.mean(D_sorted_moist[idx])
        
    return D_sorted_unique_moist


def dD_dtheta_fun(dD_dtheta, moist_1_sorted, moist1_idx_sorted, moist_1_sorted_unique):
    # Mean values of dD/dtheta with respect to the moisture(theta) at sensor 1
    dD_dtheta_sorted = dD_dtheta[moist1_idx_sorted]
    dD_dtheta_sorted_unique_moist = np.zeros_like(moist_1_sorted_unique)
    for i in range(len(moist_1_sorted_unique)):
        idx = moist_1_sorted == moist_1_sorted_unique[i]
        dD_dtheta_sorted_unique_moist[i] = np.mean(dD_dtheta_sorted[idx])
    D_integrated = integrate.cumulative_trapezoid(dD_dtheta_sorted_unique_moist, moist_1_sorted_unique, initial=0)

    return dD_dtheta_sorted_unique_moist, D_integrated

def K_fun(dk_dtheta, moist_1_sorted, moist1_idx_sorted, moist_1_sorted_unique):
    dk_dtheta_sorted_moist = dk_dtheta[moist1_idx_sorted]
    dk_dtheta_sorted_unique_moist = np.zeros_like(moist_1_sorted_unique)
    for i in range(len(moist_1_sorted_unique)):
        idx = moist_1_sorted == moist_1_sorted_unique[i]
        dk_dtheta_sorted_unique_moist[i] = np.mean(dk_dtheta_sorted_moist[idx])
    K_integrated = integrate.cumulative_trapezoid(dk_dtheta_sorted_unique_moist, moist_1_sorted_unique, initial=0)

    return K_integrated

def q_fun(D_sorted_unique_moist, K_sorted_unique_moist, dmoist_dz, moist_1_sorted, moist1_idx_sorted, moist_1_sorted_unique):
    # q(theta), use K and D sorted coreesponding to theta(moisture at midle sensor - sensor 1)
    dmoist_dz_sorted_unique_moist = np.zeros_like(moist_1_sorted_unique)
    dmoist_dz_sorted_moist = dmoist_dz[moist1_idx_sorted]
    for i in range(len(moist_1_sorted_unique)):
        idx = moist_1_sorted == moist_1_sorted_unique[i]
        dmoist_dz_sorted_unique_moist[i] = np.mean(dmoist_dz_sorted_moist[idx])
    q_sorted_unique_moist = D_sorted_unique_moist* dmoist_dz_sorted_unique_moist + K_sorted_unique_moist
    
    return q_sorted_unique_moist


