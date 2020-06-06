#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:49:58 2019

@author: axj307
"""


import numpy as np
from numpy.matlib import repmat

def Pcomb(ND, Ninteg, xinteg, winteg ):
    
    index = np.arange(Ninteg[1])
    
    for ct in range(1,ND):
        repel = index
        repsize = len(index[:])
        repwith = np.zeros([repsize, 1])
        
        for rs in range(1, Ninteg[1]):
            repwith = np.concatenate( (repwith, rs*np.ones([repsize, 1])), axis =0)
        index = np.hstack(( repmat(repel.T, 1,  Ninteg[1] ).T,  repwith ))  
    
    
    xint = np.zeros([len(index), ND] )
    wint_temp = np.zeros([len(index), ND] )
    
    if np.shape(xinteg)[1] ==1:
        index = index.reshape((len(index),1))
    for i in range(len(index)):
        for j in range(ND):
            xint[i, j] = xinteg[int(index[i,j] ), j]
            wint_temp[i, j] = winteg[int(index[i,j] ), j]
    
    wint = wint_temp[:,0]
    for i in range(1,ND):
        wint = wint*wint_temp[:,i]
    
    return(xint, wint)