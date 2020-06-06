#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:10:44 2020

@author: axj307
"""
import numpy as np


def moment_transform(bU, bL, E10, E20, E30, E40):
    
    if len(bU) == 2:
        M = np.array( [[ 2/(bU[0] -bL[0]) , 0],[0, 2/(bU[1] -bL[1])] ])
        N = -np.array([ [ (bU[0] + bL[0])/(bU[0] -bL[0]) ], [(bU[1] + bL[1])/(bU[1] -bL[1]) ] ] )
        
    elif len(bU) == 3:
        M = np.array( [[ 2/(bU[0] -bL[0]) , 0, 0],[0, 2/(bU[1] -bL[1]), 0] , [0, 0, 2/(bU[2] -bL[2])]])
        N = -np.array([ [ (bU[0] + bL[0])/(bU[0] -bL[0]) ], [(bU[1] + bL[1])/(bU[1] -bL[1]) ], [(bU[2] + bL[2])/(bU[2] -bL[2]) ] ] )
        
    
    m1=np.zeros([len(E10), 1])
    m1[:,0] = E10
    E10 = m1
    
    E11 = M@E10 + N
    # E10 = First_moment_x_CUT[ct]
    E11 = M@E10 + N
    
    # E20 = Second_moment_x_CUT[ct]
    
    # E30 = np.zeros([xdim, xdim, xdim])
    # E30[:,:,0] = np.array([ [m3r[0,0],m3r[1,0]],[m3r[1,0], m3r[2,0]]  ])
    # E30[:,:,1] = np.array([ [m3r[1,0],m3r[2,0]],[m3r[2,0], m3r[3,0]]  ])
    
    # E40 = np.zeros([xdim, xdim, xdim, xdim])
    # E40[:, :, 0, 0] = np.array([ [m4r[0,0], m4r[1,0]],[m4r[1,0], m4r[2,0]]  ])
    # E40[:, :, 0, 1] = np.array([ [m4r[1,0], m4r[2,0]],[m4r[2,0], m4r[3,0]]  ])
    # E40[:, :, 1, 0] = np.array([ [m4r[1,0], m4r[2,0]],[m4r[2,0], m4r[3,0]]  ])
    # E40[:, :, 1, 1] = np.array([ [m4r[2,0], m4r[3,0]],[m4r[3,0], m4r[4,0]]  ])    
    
    
    ##########################################################################################  2nd ORDER
    
    MM = np.tensordot(np.sum(M, axis =0), np.sum(M, axis =0), axes=0)
    MME = MM*E20
    NN1 = np.squeeze( np.tensordot(N,N, axes=0) )
    MN = np.squeeze( np.tensordot(np.sum(M, axis =0), N, axes=0))
    MNE = MN*E10
    
    E22 = MME + NN1 + MNE + np.transpose(MNE , (1,0) )
    
    
    ##########################################################################################   3rd ORDER
    
    NNN1 = np.squeeze( np.tensordot(NN1,N, axes=0))
    MMM = np.tensordot(MM,np.sum(M, axis =0), axes=0)
    MMME = MMM*E30
    ME = np.sum(M*E10, axis=1)
    MNNE  = np.squeeze( np.tensordot(NN1,  ME  , axes=0  ))
    MMNE = np.squeeze( np.tensordot( MME , N , axes=0  ))
    
    axe1 = (1, 2, 0)
    axe2 = (2, 0, 1)
    
    E33 = MMME + MMNE + np.transpose(MMNE, axe1) + np.transpose(MMNE, axe2) + MNNE + np.transpose(MNNE, axe1) + np.transpose(MNNE, axe2) + NNN1
    
    ######################################################################################################  4th ORDER
    NNNN1 = np.squeeze( np.tensordot(NNN1,N, axes=0))
    MMMME = np.squeeze( np.tensordot( MM, MM, axes=0))*E40
    MNNNE = np.squeeze( np.tensordot( ME, NNN1, axes=0))
    MMNNE = np.squeeze( np.tensordot( MME, NN1, axes=0))
    MMMNE = np.squeeze( np.tensordot( MMME, N, axes=0))
    
    # ax1 = (1, 0, 2, 3)
    # ax2 = (2, 0, 1, 3)
    # ax3 = (3, 0, 1, 2)
    # aX1 = (0, 2, 1, 3)
    # aX2 = (0, 3, 1, 2)
    # aX3 = (1, 2, 0, 3)
    # aX4 = (1, 3, 0, 2)
    # aX5 = (2, 3, 0, 1)
    
    ax1 = (0, 1, 3, 2)
    ax2 = (3, 0, 1, 2)
    ax3 = (0, 3, 1, 2)
    
    # aX1 = (0, 2, 1, 3)
    # aX2 = (0, 3, 1, 2)
    # aX3 = (1, 2, 0, 3)
    # aX4 = (1, 3, 0, 2)
    # aX5 = (2, 3, 0, 1)  
    aX1 = (0, 2, 1, 3)
    aX2 = (2, 0, 1, 3)
    aX3 = (0, 2, 3, 1)
    aX4 = (2, 0, 3, 1)
    aX5 = (2, 3, 0, 1) 
    
    _ax1 = (1, 0, 2, 3);   _ax2 = (1, 2, 0, 3);    _ax3 = ( 1, 2, 3, 0); 
    
    E44 = MMMME + NNNN1 + MMMNE + np.transpose(MMMNE, ax1) + np.transpose(MMMNE, ax2) + np.transpose(MMMNE, ax3)  \
    + MMNNE + np.transpose(MMNNE, aX1) + np.transpose(MMNNE, aX2) + np.transpose(MMNNE, aX3) + np.transpose(MMNNE, aX4) + np.transpose(MMNNE, aX5)  \
    + MNNNE + np.transpose(MNNNE, _ax1) + np.transpose(MNNNE, _ax2) + np.transpose(MNNNE, _ax3 )
    
    
    
    
    
    
    return(E11, E22, E33, E44)
    
    
