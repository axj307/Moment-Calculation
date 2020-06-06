#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 15:50:52 2019
@author: Jain

Input:
    parameters at which the value of the function g(u), T(x) and h(x) has to be calculated

Output:
    Vaues of the function g(u), T(x) and h(x)

"""
import numpy as np

########### Example Airplane with Gust
## Parameters
#from Dropbox.Research_work.UpdatedCodes.MainFolder.Example_Problems.Airplane_Gust_motion2D 
import Parameters_Airplane_gust_2Dsystem as Param
Parameters, InitialConditions, ControlParam, xdim, udim, Distribution = Param.parameters()
orderx, orderu, dt, tf, MC_n = Parameters
ax_0, bx_0, au_0, bu_0 = InitialConditions
au, bu = ControlParam


## Compute function g
def g(u):
    v = u[0]
    w = u[1]
    psi = u[2]
    
    g_val = [ -np.cos((psi))*v +w,
             np.sin((psi))*v      ]
    
    return g_val


## Compute function T
def T(x):
    
    T_val = np.zeros([len(ax_0), len(ax_0)])
    T_val[0, :] = [dt, 0]
    T_val[1, :] = [0, dt]
    
    return T_val


## Compute function h
def h(x):
    xk =x[0]
    yk =x[1]
   
    h_val = [ xk , 
             yk ]
    
    return h_val

