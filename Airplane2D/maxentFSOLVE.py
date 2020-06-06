#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 10:20:36 2019

@author: Jain


"""

import numpy as np
from numpy.matlib import repmat
import pdf_MaxEnt as pdf



def maxentFSOLVE(lam, y, M, X, W):
    
    nm = len(y)
    nq = len(W)
    
    ceq = np.zeros([nm, nq])
    
    for i in range(len(W)):
        
        ceq[:,i] = W[i]*np.prod( repmat(X[i,:],nm,1)**y , 1)*pdf.pdf_MaxEnt(X[i,:], lam, y)
        
        
    ceq= np.sum(ceq, 1) 
    ceq = ceq- M
        
    return(ceq)