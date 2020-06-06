#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 18:37:22 2019

@author: Jain

% given X(samples,states) and constant w-weight calculate the moments
% N is the the order of required moment

"""
import numpy as np
from numpy.matlib import repmat
import GenerateIndex as Gen

def Cal_moments_samples(X, w, N, Type):
    
    nx = np.size(X, 1)
    ns = len(X)
    
#    if len(w) ==1:
#        w=w.T
    
    W = repmat(w, 1, nx)
    mu = sum(W*X, 0)
    
    if N==1:
        M=mu.T
        y=np.eye(nx)
        return(y,M)
    
    if Type == 'central':
        # print('central moms')
        X = X -repmat(mu, ns,1)
    
    # elif  Type == 'raw':
    #     print('raw moms')
    

    
#############################################################################################################################################################################
    combos = Gen.GenerateIndex(nx, N+1, 'true', N+1)
    combos[ np.nonzero(combos==N+1)[0]] =0
    
    y_temp=[]
    for i in range(len(combos)):
        if sum(combos[i, :]) == N:
            y_temp.append( combos[i, :])
    
    y=np.asarray(y_temp)
    yy, yyy = np.shape(y)
    
    
#############################################################################################################################################################################

    M = np.zeros([yy,1])
    
    for i in range(yy):
        M[i] = sum(w*np.prod(X**repmat(y[i,:],ns, 1), 1) )
    
    
    return(y, M)