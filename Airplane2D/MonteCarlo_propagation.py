#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:46:10 2019

@author: Jain

Input:
    dt, tf, MC_n, xdim, au, bu
    
Output:
    MC_propagation: 1st column represents the time step, everything else represent the states

"""
import numpy as np
from sklearn.utils import shuffle
import scipy
###### Example Airplane with Gust
import Dynamics_Airplane_gust_2Dsystem as F

#### Example Dubins Model
# import Dynamics_DubinsModel_3Dsystem as F

# ########### Example pendulum problem
# import Dynamics_pendulum_3Dsys as F

def MonteCarlo_propagation(dt, tf, MC_n, ax, bx, au, bu, xdim, udim):
       
    ## Monte Carlo simulation
    print('Monte Carlo simulation')
    
    MC_propagation = np.zeros([int(MC_n*tf/dt), xdim+1])
    MC_Propagation = []
    
    i = 0
    for j in range(MC_n):
        u = np.zeros([ udim, 1])
        x = np.zeros([ xdim, 1])
        
        if bu.size ==1:
            u = np.random.uniform(au, bu, 1)
        elif au.shape == bu.shape:
            for k in range(udim):
                u[k, 0] = np.random.uniform(au[k], bu[k], 1)
        else:
            u = np.matmul(scipy.linalg.sqrtm(bu), np.random.randn(udim, 1)) + au


        if   ax.shape == bx.shape:
            for k in range(xdim):
                x[k] = shuffle(np.random.uniform(ax[k], bx[k], 1))
        else:            
            x = np.matmul(scipy.linalg.sqrtm(bx), np.random.randn(xdim, 1)) + ax 
            
        MC_propagation[j, 1:] =  ( np.array(F.h(x)) + np.matmul(F.T(x), F.g(u))).T
    MC_Propagation.append(MC_propagation[i*MC_n: i*MC_n+j, ])   
    
    for i in range(1, int(tf/dt)):
        # Printing progresssion
        if np.mod(i, 1)==0:
            print('Time', i*dt, 'sec over', tf, 'sec.')
            
        for j in range(MC_n):
            u = np.zeros([ udim, 1])
            if bu.size ==1:
                u = np.random.uniform(au, bu, 1)
            elif au.shape == bu.shape:
                for k in range(udim):
                    u[k] = shuffle(np.random.uniform(au[k], bu[k], 1))
            else:
                u = np.matmul(scipy.linalg.sqrtm(bu), np.random.randn(udim, 1)) + au
                
            MC_propagation[i*MC_n+j, 0] = i*dt
            x = MC_propagation[(i-1)*MC_n+j, 1:]
            MC_propagation[i*MC_n+j, 1:] = np.array(F.h(x)) + np.transpose(np.matmul(F.T(x), F.g(u)))
        MC_Propagation.append(MC_propagation[i*MC_n: i*MC_n+j, ])
 
    return(MC_propagation, MC_Propagation)
    
    
