#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:28:25 2019

@author: axj307
"""

import numpy as np
import numpy.polynomial.legendre as leg
import Pcomb as P

def GLeg_pts(Ninteg, xl,xu ):
#    
#    Ninteg = [5, 5]
#    xl = [-1, -1]
#    xu = [1, 1]
    
    if np.size(xl) ==1:
        ND = np.size(xl)
        xl=np.squeeze(xl)
        xu=np.squeeze(xu)
        
        xinteg = np.zeros([Ninteg[0], ND])
        winteg = np.zeros([Ninteg[0], ND])
        
        for ct in range(ND):
            [xint, wint] =  leg.leggauss(Ninteg[ct])   ## a=-1 , b=1   integrate from -1, 1
            ### Linear map from[-1,1] to [a,b]
            xint = (xl*(1-xint)+ xu*(1+xint))/2 
            wint = wint*(xu - xl)/2
            
            xinteg[:, ct] = xint
            winteg[:, ct] = wint
            
        del xint, wint
        
        [xint, wint] = P.Pcomb(ND, Ninteg, xinteg, winteg)
        
        wint = wint / sum(wint)        
    
    
    else:
        ND = np.size(xl)
        xl=np.squeeze(xl)
        xu=np.squeeze(xu)
        
        xinteg = np.zeros([Ninteg[0], ND])
        winteg = np.zeros([Ninteg[0], ND])
        
        for ct in range(ND):
            [xint, wint] =  leg.leggauss(Ninteg[ct])   ## a=-1 , b=1   integrate from -1, 1
            ### Linear map from[-1,1] to [a,b]
            xint = (xl[ct]*(1-xint)+ xu[ct]*(1+xint))/2 
            wint = wint*(xu[ct] - xl[ct])/2
            
            xinteg[:, ct] = xint
            winteg[:, ct] = wint
            
        del xint, wint
        
        [xint, wint] = P.Pcomb(ND, Ninteg, xinteg, winteg)
        
        wint = wint / sum(wint)
    
    return(xint, wint)    
        
    
