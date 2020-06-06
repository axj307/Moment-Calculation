#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 22:33:10 2019

@author: Jain



"""

import numpy as np
from numpy.matlib import repmat


def transform_domain(Xb, bl, bu, Bl, Bu):
    
    nx = np.size(Xb, 1)
    ns = len(Xb)
    
    rb=0.5*(bl+bu)
    #shifting to 0 mean
    X0 = Xb - repmat(rb, ns, 1)
    
    #scaling the points
    db=bu-bl
    dB=Bu-Bl
    
    d = dB/db

    Xs = repmat(d, ns,1)*X0

    #shifting the mean

    rB=0.5*(Bl+Bu);
    #% rmu=rB-rb;

    XB=Xs+repmat(rB,ns,1)
    
    return(XB, d)
