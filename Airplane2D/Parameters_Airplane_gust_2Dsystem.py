#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 09:31:03 2019
@author: Jain

Parameters:
    au:  Mean / lower bound for the control input
    bu: Covairance / upper bound for the control input 

"""
import numpy as np

## Here are set constant parameters
def parameters():
    
#    ## General parameters
    dt = 0.5                          # [s]
    tf = 11                 # [s]
    MC_n = 100000

    orderx = 8
    orderu = 8

#    # Initial Mean and Covariance for Gaussian distribution or lower and upper bounds for the uniform distribution for states 
#    #uniform states
    ax_0 = np.zeros([2, 1]) 
    bx_0 = np.zeros([2, 1]) 
    
#   #Mean and Covariance for Gaussian distribution or the lower and upper bounds for the uniform distribution for control input
    #Bound for first control input
#    #uniform control
    # # Define extra parameters according to your problem here
    apsi = np.deg2rad(-10)
    bpsi = np.deg2rad(10)
    
    au_0 = np.array( [[9.5], [-0.3], [apsi]])
    bu_0 = np.array( [[10.5], [0.3], [bpsi]])
#    
#    au = np.zeros([int(tf/dt), 2]) 
#    bu = np.zeros([int(tf/dt), 2]) 
#    au[:, 0] = au_0[0]*np.ones(int(tf/dt)) 
#    au[:, 1] = au_0[1]*np.ones(int(tf/dt)) 
#    bu[:, 0] = bu_0[0]*np.ones(int(tf/dt)) 
#    bu[:, 1] = bu_0[1]*np.ones(int(tf/dt)) 
#    
#    #Gaussian States
#    ax_0 = np.ones([2,1]) 
#    bx_0 = np.ones([2,2]) 
#
#    # Initial Gaussian Control
#    au_0 = np.ones([2, 1])
#    bu_0 = 3**2*np.ones([2, 2])
    
    
    if ax_0.shape == bx_0.shape:
        mx, nx = bx_0.shape
        xdim = mx
        stateDistribution = 'uniform'
        
    else:
        mx, nx = bx_0.shape
        xdim = mx
        stateDistribution = 'gaussian'
        
    if au_0.shape == bu_0.shape:
        mu, nu = au_0.shape
        udim = mu
        controlDistribution = 'uniform'
        
    else:
        mu, nu = bu_0.shape
        udim = mu
        controlDistribution = 'gaussian'
    
    # Assuming a constant Control profile
    au = au_0
    bu = bu_0
    
    
    InitialConditions = [ax_0, bx_0, au_0, bu_0]
    Parameters = [ orderx, orderu, dt, tf, MC_n] 
    ControlParam = [au, bu]
    Distribution = [stateDistribution, controlDistribution]
    
    return (Parameters, InitialConditions, ControlParam, xdim, udim, Distribution)
