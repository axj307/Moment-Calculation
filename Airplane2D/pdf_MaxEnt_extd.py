#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:07:33 2019
@author: jain


"""

import numpy as np
from numpy.matlib import repmat

def pdf_MaxEnt_extd(x, lam, y, xl, xu, A):
    
    P = np.zeros([1 ])
    ns = np.shape(x)[1]
    nm = np.size(lam)
    
    
    for i in range(ns):
        if sum( np.sign(x -xl)[0] ) == np.size(xl) and sum( np.sign(xu -x)[0] ) == np.size(xl) :
            P[i] = np.exp( lam@A@np.prod(repmat(x.T,nm,1)**y , 1) )
            
        
#    p = np.exp( sum( np.multiply(np.prod(repmat(x,nm,1)**y , 1), lam) ))


    return(P)
    
    
    
#for i=1:1:size(x,1)
#    if sum(sign(x(i,:)-xl))==length(xl) && sum(sign(xu-x(i,:)))==length(xl)
#        P(i)=exp(lam'*A*prod(repmat(x,nm,1).^y,2));
#    else
#        P(i)=0;
#        
#    end
#    
#end