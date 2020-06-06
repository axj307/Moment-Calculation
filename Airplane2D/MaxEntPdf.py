#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 09:20:03 2019

@author: Jain

 % xl and xu are the lower and uppre bounds for the 
% y contains all the combinations for the moments
% M starts from the second moment

"""

import numpy as np
from numpy.matlib import repmat
import GLeg_pts as GL
from scipy.optimize import least_squares
from scipy.optimize import root
from scipy.optimize import minimize
import pdf_MaxEnt as pdf


def MaxEntPdf(y, M, xl, xu, lam0, method):
    
    # ns = np.size(y,1)
    # nm = len(y)
    
    if method == 'GL':
        Ninteg = np.squeeze( 10*np.ones(np.shape(xl)) ).astype(int)  #### 10*np.ones((3)).astype(int)
        X, W = GL.GLeg_pts(  Ninteg, xl, xu)
        
        
    elif method == 'GH':
        print('GH method not defined')
    
    def PME_Costfunct(lam):
        
        nm = len(y)
        nq = len(W)
        
        ceq = np.zeros([nm, nq])
        
        for i in range(len(W)):
            ceq[:,i] = W[i]*np.prod( repmat(X[i,:],nm,1)**y , 1)*pdf.pdf_MaxEnt(X[i,:], lam, y)
        
        ceq= np.sum(ceq, 1) - M
        
        # ceq= sum( ceq**2)
        
        
        return(np.squeeze(ceq))

    # solution = minimize(  PME_Costfunct , lam0 ,  method='BFGS', tol=1e-12 , options={'disp':True, 'maxiter':1000000}) 
#    lam=solution 


    # solution = root(  PME_Costfunct , lam0 ,  method='lm', tol=1e-15 , options={'ftol':1e-17, 'xtol' :1e-17, 'gtol' :1e-17, 'maxiter':1000000}) 
    ### lam=solution      
        
    solution = least_squares(  PME_Costfunct , lam0 ,  method='lm', ftol=1e-17, xtol =1e-15 , gtol=1e-14)  #,  method='lm' 
    ### lam=solution

    
    return(y, solution)
    