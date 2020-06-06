#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 09:38:52 2020

@author: axj307


Inputs: 
    ax: Mean \ lower bound of state                 --- can be (nx1)  or (1xn)              float
    bx: Covariance \ upper bound of state           --- can be (nxn), (nx1)  or (1xn)       float 
    au: Mean \ lower bound of control               --- can be (mx1)  or (1xm)              float
    bu: Covariance \ upper bound of control         --- can be (mxm), (mx1)  or (1xm)       float
    orderx: accuracy upto  4/6/8th order for state                                          integer
    orderu: accuracy upto  4/6/8th order for state                                          integer
    
    
Output:
    E1 : 1st order Statistical moment computed through CUT method                           
    E2 : 2nd order Statistical moment computed through CUT method
#    E3 : 3rd order Statistical moment computed through CUT method
    xdim: dimension of the states


ax = np.zeros([3,1])
bx = np.diag([1,1,1])
au = np.ones([2,1])
bu = np.diag([1,1])

ax = np.ones([2,1])
bx = np.diag([1,1])
au = np.ones([2,1])
bu = np.diag([1,1])

ax = np.ones([6,1])
bx = np.ones([6,6]) + np.diag([1,1,1,1,1,1])
au = np.ones([2,1])
bu = np.zeros([2,1]) +np.diag([1,1])
"""
import numpy as np
from numpy import transpose as trans
import scipy
from sklearn.utils import shuffle
#from Dropbox.Research_work.UpdatedCodes.Functionfiles 
import cut_points_gaussian as CUTgaussian
#from Dropbox.Research_work.UpdatedCodes.Functionfiles 
import cut_points_uniform  as CUTuniform
import GLeg_pts as GL

##### Example Airplane with Gust
import Dynamics_Airplane_gust_2Dsystem as F

##### Example Dubins Model
#import Dynamics_DubinsModel_3Dsystem as F
# import Dynamics_pendulum_3Dsys as F


def moment_Next(ax, bx, au, bu, method, xdim, udim,Distribution, MC_n, orderx = 8, orderu = 8):
    
    if method =='CUT':
        if ax.shape == bx.shape:
            mx, nx = bx.shape
            # if mx == 1:
            #     xdim = nx
            #     mat_x = CUTuniform.cut_points_uniform(xdim, orderx)
            #     x = np.zeros([len(mat_x), 1, xdim])
            #     Nx = len(x[:, 0])
            #     for k in range(xdim):
            #         x[:, :, k] = mat_x[:,k]*(bx[k]-ax[k])/2 + (bx[k]+ax[k])/2
            #     x = np.squeeze(x)
            #     wx = np.zeros([Nx, 1])
            #     wx = np.array(mat_x[:, xdim])
                      
            
            if nx == 1:
                xdim = mx
                mat_x = CUTuniform.cut_points_uniform(xdim, orderx)
        
        
                x = np.zeros([len(mat_x), 1, xdim])
                Nx = len(x[:, 0])
                for k in range(xdim):
                    x[:, :, k] = mat_x[:,k]*(bx[k]-ax[k])/2 + (bx[k]+ax[k])/2
                x = np.squeeze(x)
                wx = np.zeros([Nx, 1])
                wx = np.array(mat_x[:, xdim])
                   
                
        else :
            mx, nx = bx.shape
            xdim = mx
            mat_x = CUTgaussian.cut_points_gaussian(xdim, orderx)
            x = np.matmul(mat_x[:, 0:xdim], scipy.linalg.sqrtm(bx)) + trans(ax)
            x= np.array(x)
            Nx = len(x[:, 0])
            wx = np.zeros([Nx, 1])
            wx = np.array(mat_x[:, xdim])
            
        
        if au.shape == bu.shape:
            if au.shape == ():
                u, wutemp = GL.GLeg_pts( [orderu,orderu], au, bu )
                Nu = len(u)
                wu = np.zeros([len(wutemp), 1])
                wu[:,0] = wutemp
            
            else:
                mu, nu = au.shape
                # if mu == 1:
                #     udim = nu
                #     mat_u = CUTuniform.cut_points_uniform(udim, orderu)
                #     u = np.zeros([len(mat_u), 1, udim])
                #     Nu = len(u[:, 0])
                #     for k in range(udim):
                #         u[:, :, k] = mat_u[:,k]*(bu[k]-au[k])/2 + (bu[k]+au[k])/2
                #     u = np.squeeze(u)
                #     wu = np.zeros([Nu, 1])
                #     wu = np.array(mat_u[:, udim])
                    
                        
                if nu == 1:
                    udim = mu
                    mat_u = CUTuniform.cut_points_uniform(udim, orderu)
                    u = np.zeros([len(mat_u), 1, udim])
                    Nu = len(u[:, 0])
                    for k in range(udim):
                        u[:, :, k] = mat_u[:,k]*(bu[k]-au[k])/2 + (bu[k]+au[k])/2
                    u = np.squeeze(u)
                    wu = np.zeros([Nu, 1])
                    wu = np.array(mat_u[:, udim])
            
        else :
            mu, nu = bu.shape
            udim = mu
            mat_u = CUTgaussian.cut_points_gaussian(udim, orderu)
            u = np.matmul(mat_u[:, 0:udim],scipy.linalg.sqrtm(bu)) +  trans(au)
            u = np.array(u)
            Nu = len(u[:, 0])
            wu = np.zeros([Nu, 1])
            wu = np.array(mat_u[:, udim])
            
    if method =='MC':  
        Nx = MC_n
        Nu = MC_n
        if ax.shape == bx.shape:
            x = np.zeros([ MC_n, xdim])
            for k in range(xdim):
                x[:, k] = shuffle(np.random.uniform(ax[k], bx[k], MC_n))
            wx= 1/MC_n*np.ones([MC_n,1])
            
            
        else:
            ###GAUSSIAN 
            x = trans( np.matmul( scipy.linalg.sqrtm(bx), np.random.randn(xdim, MC_n)) + ax )
            wx= 1/MC_n*np.ones([MC_n,1])
        
        if au.shape == bu.shape:
            u = np.zeros([ MC_n, udim])
            for k in range(udim):
                if au.size ==1:
                    u = np.random.uniform(au, bu, MC_n)
                
                else:
                    u[:, k] = shuffle(np.random.uniform(au[k], bu[k], MC_n))
                
            wu= 1/MC_n*np.ones([MC_n,1])
        
        else:
            
            u = trans( np.matmul( scipy.linalg.sqrtm(bu), np.random.randn(udim, MC_n)) + au )
            wu= 1/MC_n*np.ones([MC_n,1])    
        
    # Compute functions h, g, T
    h = np.zeros([Nx, xdim])
    g = np.zeros([Nu, xdim])
    T = np.zeros([Nx, xdim, xdim])
    for i in range(Nx):
        h[i, :] = F.h(x[i])
        T[i, :, :] = F.T(x[i])
    for i in range(Nu):
        g[i, :] = F.g(u[i])     
        
    # h = h - repmat(np.mean(h,0) , len(h), 1 )
    
    wx_h = wx*h
    wu_g = wu*g
    wx_T  = np.array( [wx[i]*T[i, :, :] for i in range(Nx)] )
    wx_hh = np.array( [np.outer(wx_h[i, :], h[i, :]) for i in range(Nx)])
    # wx_hh1 = np.array( [ wx[i]*np.outer(h[i, :], h[i, :]) for i in range(Nx)])
    
    wu_gg = np.array( [np.outer(wu_g[i, :], g[i, :]) for i in range(Nu)])
    wx_hT = np.array( [[wx_h[i, j]*T[i, :, :] for j in range(xdim)] for i in range(Nx)])
    wx_TT = np.array( [[[wx_T[i, j, k]*T[i, :, :] for k in range(xdim)] for j in range(xdim)] for i in range(Nx)])
    wx_hhh = np.array( [[wx_hh[i, :, :]*h[i, j] for j in range(xdim)] for i in range(Nx)])
    wu_ggg = np.array( [[wu_gg[i, :, :]*g[i, j] for j in range(xdim)] for i in range(Nu)])
    wx_hhT = np.array( [[[wx_hh[i, j, k]*T[i, :, :] for k in range(xdim)] for j in range(xdim)] for i in range(Nx)])
    wx_hTT = np.array( [[[[wx_hT[i, j, k, l]*T[i, :, :] for l in range(xdim)] for k in range(xdim)] for j in range(xdim)] for i in range(Nx)])
    wx_TTT = np.array( [[[[[wx_TT[i,j,k,l,m]*T[i, :, :] for m in range(xdim)] for l in range(xdim)] for k in range(xdim)] for j in range(xdim)] for i in range(Nx)])
    wx_hhhh = np.array( [[[wx_hhh[i, j, :, :]*h[i, k] for k in range(xdim)] for j in range(xdim)] for i in range(Nx)])
    wu_gggg = np.array( [[[wu_ggg[i, j, :, :]*g[i, k] for k in range(xdim)] for j in range(xdim)] for i in range(Nu)])
    wx_hhhT = np.array( [[[[wx_hhh[i, j, k, l]*T[i, :, :] for l in range(xdim)] for k in range(xdim)] for j in range(xdim)] for i in range(Nx)])
    wx_hhTT = np.array( [[[[[wx_hhT[i, j, k, l, m]*T[i, :, :] for m in range(xdim)] for l in range(xdim)] for k in range(xdim)] for j in range(xdim)] for i in range(Nx)])
    wx_hTTT = np.array( [[[[[[wx_hTT[i, j, k, l, m, n]*T[i, :, :]  for n in range(xdim)] for m in range(xdim)] for l in range(xdim)] for k in range(xdim)] for j in range(xdim)] for i in range(Nx)])
    wx_TTTT = np.array( [[[[[[[wx_TTT[i, j, k, l, m, n, o]*T[i, :, :] for o in range(xdim)]  for n in range(xdim)] for m in range(xdim)] for l in range(xdim)] for k in range(xdim)] for j in range(xdim)] for i in range(Nx)])


        
    # Compute first moment
    Eh = np.sum(wx_h , 0)
    ETg = np.sum(wx_T, 0)@np.sum(wu_g, 0)
        
    # Compute second moment
    Ehh = np.sum(wx_hh , 0)
    ETTgg = np.sum( wx_TT , (0, 2, 4))*np.sum(wu_gg, 0)
    EhTg = np.sum(wx_hT , (0, 3))*np.sum(wu_g, 0)

    # Compute third moment
    Ehhh = np.sum(wx_hhh, 0)
    ETTTggg = np.sum( wx_TTT , (0, 2, 4, 6))*np.sum(wu_ggg, 0)
    EhhTg = np.sum(wx_hhT , (0, 4))*np.sum(wu_g, 0)
    EhTTgg = np.sum(wx_hTT , (0, 3, 5))*np.sum(wu_gg, 0)
    
    #Compute fourth moment
    Ehhhh = np.sum(wx_hhhh, 0)
    ETTTTgggg = np.sum( wx_TTTT , (0, 2, 4, 6, 8))*np.sum(wu_gggg, 0)
    EhhhTg = np.sum(wx_hhhT , (0, 5))*np.sum(wu_g, 0)
    EhhTTgg = np.sum(wx_hhTT , (0, 4, 6))*np.sum(wu_gg, 0)
    EhTTTggg = np.sum(wx_hTTT , (0, 3, 5, 7))*np.sum(wu_ggg, 0)
    
    
    # Building first moment    
    E1 = Eh + ETg
    
    # Building second moment
    E2_raw = Ehh + EhTg + trans( EhTg, (1,0)) + ETTgg
    
    E1_2 = np.outer(E1, E1)
    E2 = E2_raw - E1_2

    # Building third moment    
    axe1 = (1, 2, 0);    axe2 = (2, 0, 1)
    E3_raw = Ehhh + EhhTg + trans(EhhTg, axe1) + trans(EhhTg, axe2) + EhTTgg + trans(EhTTgg, axe1) +  trans(EhTTgg, axe2) + ETTTggg
    # Building tensor of mu^3    
    E1_2 = np.outer(E1, E1)
    E1_3 = np.tensordot(E1, E1_2, axes=0)
    E1E2_raw = np.tensordot(E1, E2_raw, axes=0)
    E3 = E3_raw + 2*E1_3 - E1E2_raw - trans( E1E2_raw, axe1) - trans( E1E2_raw, axe2)
    
    # Building fourth moment
    ax1 = (1, 2, 3, 0);     ax2 = (0, 3, 2, 1);    ax3 = ( 3, 0, 1, 2);     
    aX1 = (2, 0, 1, 3);    aX2 = (2,0, 3, 1);    aX3 = (0, 2, 1, 3);    aX4 = (0, 2, 3, 1);    aX5 = (2, 3, 0, 1)
    _ax1 = (1, 2, 3, 0);   _ax2 = (1, 0, 3, 2);    _ax3 = ( 1, 2, 0, 3);     
    
    
    E4_raw = Ehhhh   + EhhhTg + trans(EhhhTg, ax1) + trans(EhhhTg, ax2) + trans(EhhhTg, ax3) \
    + EhhTTgg + trans(EhhTTgg, aX1) + trans(EhhTTgg, aX2) + trans(EhhTTgg, aX3) + trans(EhhTTgg, aX4) +  trans(EhhTTgg, aX5) \
    + EhTTTggg +  trans(EhTTTggg, _ax1) + trans(EhTTTggg, _ax2) + trans(EhTTTggg, _ax3) + ETTTTgggg        
    
    E1_4 = np.tensordot(E1, E1_3, axes=0)
    E1E3_raw = np.squeeze( np.tensordot(E1, E3_raw, axes=0) )
    E1_2_E2_raw = np.tensordot(E1_2, E2_raw, axes=0)
    
    E4 = E4_raw -3*E1_4 - E1E3_raw - trans(E1E3_raw, ax1) - trans(E1E3_raw, ax2) - trans(E1E3_raw, ax3)  \
    - E1_2_E2_raw - trans(E1_2_E2_raw, aX1) - trans(E1_2_E2_raw, aX2) - trans(E1_2_E2_raw, aX3) - trans(E1_2_E2_raw, aX4) - trans(E1_2_E2_raw, aX5)
    
    
    return(E1, E2_raw, E2, E3_raw, E3, E4_raw, E4)
     
    
