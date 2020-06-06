#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:40:05 2019

@author: Jain
"""
import numpy as np
# import  moment_1234_next as Mn

import  moment_1234_Next as MN


def moment_propagation(tf, dt, InitialConditions, ControlParam, Distribution, method, MC_n, xdim, udim, orderx , orderu):
    
    ax_0, bx_0, au_0, bu_0 = InitialConditions
    au, bu = ControlParam
    stateDistribution, controlDistribution = Distribution
    
    ## Building time history of First and Second moments
    First_moment  = []
    First_moment_array = np.zeros([int(tf/dt), xdim+1])
    Second_moment_raw = []
    Second_moment_central = []
    Sigma3_array = np.zeros([int(tf/dt), xdim, xdim])
    Third_moment_central = []
    Third_moment_raw = []
    Fourth_moment_central = []
    Fourth_moment_raw = []
    First_moment_u = []
    Second_moment_u = []
    
    if stateDistribution == 'uniform':
    ### Propagating the first moment moments
        i=0         ## first time step
        # E1_x, E2_non_central_x, E2_x, E3_x, E4_x = Mn.moment_next(ax_0, bx_0, au_0, bu_0, method, xdim, udim, Distribution, MC_n, orderx, orderu)         ##Initial mean and covariance
        
        # E1_x, E2_non_central_x, E2_x, E3_x, E4_x = MN.moment_Next(ax_0, bx_0, au_0, bu_0, method, xdim, udim, Distribution, MC_n, orderx, orderu)         ##Initial mean and covariance
        E1, E2_raw, E2, E3_raw, E3, E4_raw, E4 = MN.moment_Next(ax_0, bx_0, au_0, bu_0, method, xdim, udim, Distribution, MC_n, orderx, orderu)         ##Initial mean and covariance
        
        First_moment.append(E1)
        Second_moment_raw.append(E2_raw)
        Second_moment_central.append(E2)
        Third_moment_raw.append(E3_raw)
        Third_moment_central.append(E3)
        Fourth_moment_raw.append(E4_raw)
        Fourth_moment_central.append(E4)

        
        First_moment_u.append(au_0)
        Second_moment_u.append(bu_0)
        
        
        First_moment_array[i, 0] = i*dt
        First_moment_array[i, 1:] =  np.transpose(E1)
        Sigma3_array[i, :, 0] =np.transpose(E1) + 3*np.sqrt(np.diag(E2))
        Sigma3_array[i, :, 1] = np.transpose(E1) - 3*np.sqrt(np.diag(E2))
        
        ### Propagating the moments
        print('Moment propagation using ' + str(method) + ' method')
        for i in range(1, int(tf/dt)):
            # Printing progresssion
            if np.mod(i, 1)==0:
                print('Time', i*dt, 'sec over', tf, 'sec.')
            
            E1, E2_raw, E2, E3_raw, E3, E4_raw, E4 = MN.moment_Next(First_moment[-1], Second_moment_central[-1], First_moment_u[-1], Second_moment_u[-1], method, xdim, udim, Distribution, MC_n, orderx, orderu)         ##Initial mean and covariance
        
            
            # E1_x, E2_non_central_x, E2_x, E3_x, E4_x = Mn.moment_next( First_moment_x[-1], Second_moment_vec_x[-1], First_moment_u[-1], Second_moment_u[-1], method, xdim, udim, Distribution, MC_n, orderx, orderu)
            
            # Appending to lists and arrays
            First_moment.append(E1)
            Second_moment_raw.append(E2_raw)
            Second_moment_central.append(E2)
            Third_moment_central.append(E3)
            Third_moment_raw.append(E3_raw)
            Fourth_moment_central.append(E4)
            Fourth_moment_raw.append(E4_raw)
            
            First_moment_array[i, 0] = i*dt
            First_moment_array[i, 1:] =  np.transpose(E1)
            Sigma3_array[i, :, 0] = np.transpose(E1) + 3*np.sqrt(np.diag(E2_raw))
            Sigma3_array[i, :, 1] = np.transpose(E1) - 3*np.sqrt(np.diag(E2_raw))
            
            # Calculating next control
            au_u_next = np.zeros([udim, 1])
            au_u_next = au
            First_moment_u.append(au_u_next)
            
    else:
        First_moment.append(ax_0)
        Second_moment_central.append(bx_0)
        Second_moment_raw.append( bx_0 + np.outer(ax_0, ax_0) )
        Third_moment_central.append(np.zeros([xdim, xdim, xdim]))
        Third_moment_raw.append(np.zeros([xdim, xdim, xdim]))
        Fourth_moment_central.append(np.zeros([xdim, xdim, xdim, xdim]))
        Fourth_moment_raw.append(np.zeros([xdim, xdim, xdim, xdim]))
        
        First_moment_u.append(au_0)
        Second_moment_u.append(bu_0)
        First_moment_array[0, 1:] = np.transpose(ax_0 )
        Sigma3_array[0,:, 0] = np.transpose(ax_0) + 3*np.sqrt(np.diag(bx_0))
        Sigma3_array[0, :, 1] = np.transpose(ax_0) - 3*np.sqrt(np.diag(bx_0))
    
        ### Propagating the moments
        print('Moment propagation using ' + str(method) + ' method')
        for i in range(1, int(tf/dt)):
            # Printing progresssion
            if np.mod(i, 1)==0:
                print('Time', i*dt, 'sec over', tf, 'sec.')
    
                 # Calculating next moment
            E1, E2_raw, E2, E3_raw, E3, E4_raw, E4 = MN.moment_Next(First_moment[-1], Second_moment_central[-1], First_moment_u[-1], Second_moment_u[-1], method, xdim, udim, Distribution, MC_n, orderx, orderu)         ##Initial mean and covariance
        
            # E1_x, E2_non_central_x, E2_x, E3_x, E4_x = Mn.moment_next( First_moment_x[-1], Second_moment_vec_x[-1], First_moment_u[-1], Second_moment_u[-1], method, xdim, udim, Distribution, MC_n,  orderx, orderu)
            
            
            # Appending to lists and arrays
            First_moment.append(E1)
            Second_moment_raw.append(E2_raw)
            Second_moment_central.append(E2)
            Third_moment_central.append(E3)
            Third_moment_raw.append(E3_raw)
            Fourth_moment_central.append(E4)
            Fourth_moment_raw.append(E4_raw)
                        
            First_moment_array[i, 0] = i*dt
            First_moment_array[i, 1:] =  np.transpose(E1)
            Sigma3_array[i, :, 0] =np.transpose(E1) + 3*np.sqrt(np.diag(E2))
            Sigma3_array[i, :, 1] = np.transpose(E1) - 3*np.sqrt(np.diag(E2))
            
            # Calculating next control
            au_u_next = np.zeros([udim, 1])
            au_u_next = au
            First_moment_u.append(au_u_next)
            
            
    return(First_moment, First_moment_array, Second_moment_raw, Second_moment_central, Sigma3_array, Third_moment_raw, Third_moment_central, Fourth_moment_raw, Fourth_moment_central)