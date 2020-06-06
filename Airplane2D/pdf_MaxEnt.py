#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 13:58:40 2019

@author: jain


"""

import numpy as np
from numpy.matlib import repmat

def pdf_MaxEnt(x, lam, y):
#    
    if np.shape(x)[0] > 1:
        x = x.T
        
    if np.shape(lam)[0] > 1:
        lam = lam.T
        
#    ns = np.shape(x)[1]
    nm = np.size(lam)
    
    p = np.exp( sum( np.multiply(np.prod(repmat(x,nm,1)**y , 1), lam) ))


    return(p)