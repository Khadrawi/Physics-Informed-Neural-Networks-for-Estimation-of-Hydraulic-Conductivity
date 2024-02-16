# -*- coding: utf-8 -*-
import torch
from numpy.core.numeric import NaN
import torch.nn as nn
import numpy as np
import random
import math as math
from tqdm import trange
import copy
from analysis_plot_functions import *
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, input_size, feature_map, network_AutoReg_index,delta_z, bias=False, expansion = 2):
        super().__init__()
        # K['last'] = nn.Linear(len_feature_map),1, bias=bias)
        self.delta_z =delta_z
        
        self.network_AutoReg_index = network_AutoReg_index
        self.out_features = 3 # len(feature_map) -1
        self.linear1_1 = nn.Linear(input_size, input_size*expansion, bias=bias) # +2 for moist_0 and +2 moist_2 as aditional inputs
        self.linear1_2 = nn.Linear(input_size*expansion, input_size*expansion, bias=bias) # +2 for moist_0 and +2 moist_2 as aditional inputs
        self.linear1_skip = nn.Linear(input_size, input_size*expansion, bias=bias) # +2 for moist_0 and +2 moist_2 as aditional inputs
        self.linear2_1 = nn.Linear(input_size*expansion, input_size*expansion, bias=bias) # +2 for moist_0 and +2 moist_2 as aditional inputs
        self.linear2_2 = nn.Linear(input_size*expansion, input_size*expansion, bias=bias) # +2 for moist_0 and +2 moist_2 as aditional inputs
        self.linear3_1 = nn.Linear(input_size*expansion, input_size*expansion, bias=bias) # +2 for moist_0 and +2 moist_2 as aditional inputs
        self.linear3_2 = nn.Linear(input_size*expansion, input_size*expansion, bias=bias) # +2 for moist_0 and +2 moist_2 as aditional inputs
        self.linear4= nn.Linear(input_size*expansion, self.out_features, bias=bias) # +2 for moist_0 and +2 moist_2 as aditional inputs
        self.batchnorm1_1 = nn.BatchNorm1d(input_size)
        self.batchnorm1_2 = nn.BatchNorm1d(input_size*expansion)
        self.batchnorm2_1 = nn.BatchNorm1d(input_size*expansion)
        self.batchnorm2_2 = nn.BatchNorm1d(input_size*expansion)
        self.batchnorm3_1 = nn.BatchNorm1d(input_size*expansion)
        self.batchnorm3_2 = nn.BatchNorm1d(input_size*expansion)

        # index of the D x lapl
        self.idx_D = list(feature_map.keys()).index('lapl')
        self.activation = customActivation(self.idx_D)

    def forward(self, input, feature_map, keep_theta_1):

        #Remove Theta 1 elements from input
        if keep_theta_1:
            x = input
        else:
            x = input[:,:feature_map['theta_1'][0]]
        
        x1 = self.batchnorm1_1(x)
        x1 = F.relu(x1)
        x1 = self.linear1_1(x1)
        x1 = self.batchnorm1_2(x1)
        x1 = F.relu(x1)
        x1 = self.linear1_2(x1) + self.linear1_skip(x)

        x2 = self.batchnorm2_1(x1)
        x2 = F.relu(x2)
        x2 = self.linear2_1(x2)
        x2 = self.batchnorm2_2(x2)
        x2 = F.relu(x2)
        x2 = self.linear2_2(x2) + x1

        x3 = self.batchnorm3_1(x2)
        x3 = F.relu(x3)
        x3 = self.linear3_1(x3)
        x3 = self.batchnorm3_2(x3)
        x3 = F.relu(x3)
        x3 = self.linear3_2(x3) + x2
        
        x3 = self.linear4(x3)
        w = self.activation(x3)
       
        # multiply each entry by a weight [D dD/dtheta ... ]
        dmoist_dz = ((self.delta_z[1]/sum(self.delta_z))*input[:, feature_map['dmoist_dz'][0]] + (self.delta_z[0]/sum(self.delta_z))*input[:, feature_map['dmoist_dz'][self.network_AutoReg_index+1]]).unsqueeze(1)
        # dmoist_dz_square = input[:, feature_map['dmoist_dz_square'][0]].unsqueeze(1)
        dmoist_dz_square = dmoist_dz**2
        lapl = input[:, feature_map['lapl'][0]].unsqueeze(1)
        input2 = torch.hstack([dmoist_dz, dmoist_dz_square, lapl])
        out = torch.mul(input2, w) #NOTE discarding last two inputs moist_0 and moist_2
        out = torch.sum(out, dim=1, keepdim=True)
        return out, w

class customActivation(nn.Module):
    def __init__(self, feature_map_selected_index):
        super().__init__()
        self.activation = nn.ReLU()
        self.idx = feature_map_selected_index
        
    def forward(self,x):
        # only add relu for the moisture diffusivity output which correponds to the second derivative
        out = torch.zeros_like(x)
        for i in range(x.shape[1]):
            if i == self.idx:
                out[:,i] = self.activation(x[:, self.idx])
            else:
                out[:,i] = x[:,i]

        return out

    
def validate_model(models, input_t, output_t, loss_fnc, feature_map, keep_theta_1):
    models.eval()
    with torch.no_grad():
        prediction, w = models(input_t, feature_map, keep_theta_1)
    loss = loss_fnc(prediction, output_t)

    return loss.data.item(), w

def create_model(feature_map, input_size, network_AutoReg_index, delta_z, LR, device ):
   
    model = NN(input_size, feature_map, network_AutoReg_index, delta_z, bias=True).to(device)
    optimizers = torch.optim.AdamW(model.parameters(), lr=LR)
    
    return model, optimizers

def train_model(models, optimizers, x, y, loss_func, feature_map, keep_theta_1 = False):
    loss_multiplication_factor = 1000
    optimizers.zero_grad()# clear gradients for this training step
    # !!!!!!!!!!!!!!!NOTE!!!!!!!!!!!!! Remove/keep theta 1 from network input INSIDE NETWORK ARCHITECHTURE!!!
    # !!!!!!!!!!!!!!!NOTE!!!!!!!!!!!!! NOW keeping theta 1 from network input INSIDE NETWORK ARCHITECHTURE!!!
    
    K_out, w = models(x, feature_map, keep_theta_1)
    loss1 = loss_func(K_out, y)
    loss = loss_multiplication_factor*loss1
    loss.backward()  # backpropagation, compute gradients
    optimizers.step()

    return loss

def train_models_loop(train_reps, train_set_dataloader, trainset_t, dmoist_dt_train_t, delta_z, epochs, moist_1_train,\
    dmoist_dz_1_train, feature_map, input_size, network_AutoReg_index, LR, device, keep_theta_1=False ):
    # To save model with lowest validation min from all train_reps repetitions
    loss_func = nn.MSELoss()  # handle # the target label is not one-hotted
    
    train_loss = np.zeros(epochs)
    
    train_min_all = math.inf
    K_min = None # model with smallest valid loss
    t3 = trange(0, train_reps) # Epochs
    t1 = trange(0, epochs) # Epochs
    t2 = trange(0, len(train_set_dataloader)) # Batches
    
    # Moisture sensor 1 - Plus sorting and getting unique values in order to compute mean per moisture value
    moist1_idx_sorted = np.argsort(moist_1_train)
    moist_1_sorted = moist_1_train[moist1_idx_sorted]
    moist_1_sorted_unique = np.unique(moist_1_sorted)

    K_all = np.zeros([train_reps, moist_1_sorted_unique.shape[0]])
    D_all = np.zeros([train_reps, moist_1_sorted_unique.shape[0]])
    q_all = np.zeros([train_reps, moist_1_sorted_unique.shape[0]])
    dD_dtheta_all = np.zeros([train_reps, moist_1_sorted_unique.shape[0]])
    D_integrated_all = np.zeros([train_reps, moist_1_sorted_unique.shape[0]])

    for j in range(train_reps):
        t1.reset()
        train_min = math.inf
        RRE_Net, optimizers = create_model(feature_map, input_size, network_AutoReg_index, delta_z, LR, device)
        for epoch in range(epochs):
            RRE_Net.train()
            t2.reset()
            for step, (x,y) in enumerate(train_set_dataloader):
                
                loss = train_model(RRE_Net, optimizers, x, y, loss_func, feature_map, keep_theta_1)
                t2.set_description('Epoch: %04d | num: %04d | train fold loss: %.5e | full train loss: %.5e' % (epoch, step, loss.data.item(), (train_loss[epoch-1] if epoch>0 else NaN)))
                t2.update(1)

            t1.update(1)
            
            train_loss[epoch], weights_train = validate_model(RRE_Net, trainset_t, dmoist_dt_train_t, loss_func, feature_map, keep_theta_1)
            if (train_loss[epoch] < train_min):
                train_min = train_loss[epoch]
                RRE_Net_min = copy.deepcopy(RRE_Net) 
                weights_train_min = weights_train
        #---Save model with lowest Valid loss---
        if train_min < train_min_all:
            train_min_all = train_min
            RRE_Net_min_all = copy.deepcopy(RRE_Net_min)
        t3.set_description('train loss min all: %.5e' % ( train_min_all))
        t3.update(1) 

        #---Analyze K, D---                
        weight_names = ["-dK/dtheta", "dD/dtheta", "D"] # Wrote this just to remeber the order of the outputs
        minus_dk_dtheta_id = weight_names.index('-dK/dtheta')
        dk_dtheta = -1*weights_train_min[:, minus_dk_dtheta_id].numpy()
        D_id = weight_names.index('D')
        D = weights_train_min[:, D_id].numpy()
        
        dD_dtheta_id = weight_names.index('dD/dtheta')
        dD_dtheta = weights_train_min[:, dD_dtheta_id].numpy()
        
        D_all[j] = D_fun(D, moist_1_sorted, moist1_idx_sorted, moist_1_sorted_unique)
        dD_dtheta_all[j], D_integrated_all[j] = dD_dtheta_fun(dD_dtheta, moist_1_sorted, moist1_idx_sorted, moist_1_sorted_unique)
        K_all[j] = K_fun(dk_dtheta, moist_1_sorted, moist1_idx_sorted, moist_1_sorted_unique)
        q_all[j] = q_fun(D_all[j], K_all[j], dmoist_dz_1_train, moist_1_sorted, moist1_idx_sorted, moist_1_sorted_unique)

    t1.close()
    t2.close()
    t3.close()

    return RRE_Net_min_all, D_all, dD_dtheta_all, D_integrated_all, K_all, moist_1_sorted_unique