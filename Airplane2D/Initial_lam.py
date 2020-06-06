#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 12:19:36 2019

@author: Jain


"""

import numpy as np
import sympy as sym
from sympy.matrices import Matrix

def Initial_lam(y1, y2, m):
    nx = np.size(y1, 1)
    
    if nx ==2:
        xs = sym.Symbol('x')
        ys = sym.Symbol('y')
        Sigma = Matrix( ( [ ( m[3] - m[1]**2), (m[4] - m[1]*m[2]) ],    [ (m[4] - m[1]*m[2]) , (m[5] - m[2]**2)   ] ))
        Sigma_inv = Sigma.inv( )
        pdf2c = (-1/2)*Matrix([xs - m[1] , ys - m[2] ]).T*Sigma_inv*Matrix([xs - m[1], ys - m[2]  ])        
        pdf = pdf2c[0].as_poly()
        
        if np.sum(Sigma_inv) == np.trace(Sigma_inv):
            lam0 = np.zeros([len(m),1])
            lam0[0] = pdf.coeffs()[4]
            lam0[1] = pdf.coeffs()[1]
            lam0[2] = pdf.coeffs()[3]
            lam0[3] = pdf.coeffs()[0]
            lam0[5] = pdf.coeffs()[2]   
            
        else:
            lam0 = np.zeros([len(m),1])
            lam0[0] = pdf.coeffs()[5]
            lam0[1] = pdf.coeffs()[2]
            lam0[2] = pdf.coeffs()[4]
            lam0[3] = pdf.coeffs()[0]
            lam0[4] = pdf.coeffs()[1]
            lam0[5] = pdf.coeffs()[3]   
        
    if nx ==3:
        xs = sym.Symbol('x')
        ys = sym.Symbol('y')
        zs = sym.Symbol('z')
        
        Sigma = Matrix( ( [ ( m[4] - m[1]**2), (m[5] - m[1]*m[2]), (m[7] - m[1]*m[3]) ],    [ (m[5] - m[1]*m[2]) , (m[6] - m[2]**2)  , (m[8] - m[2]*m[3] ) ], [ (m[7] - m[1]*m[3]) ,  (m[8] - m[2]*m[3] )  , (m[9] - m[3]**2 ) ] ))
        Sigma_inv = Sigma.inv( )
    
        pdf2c = (-1/2)*Matrix([xs - m[1] , ys - m[3], zs - m[6]]).T*Sigma_inv*Matrix([xs - m[1], ys - m[3] , zs - m[6]])        
        pdf = pdf2c[0].as_poly()
                
        lam0 = np.zeros([len(m),1])
        lam0[0] = pdf.coeffs()[9]
        lam0[1] = pdf.coeffs()[3]
        lam0[2] = pdf.coeffs()[6]
        lam0[3] = pdf.coeffs()[8]
        lam0[4] = pdf.coeffs()[0]
        lam0[5] = pdf.coeffs()[1]   
        lam0[6] = pdf.coeffs()[4]   
        lam0[7] = pdf.coeffs()[2]   
        lam0[8] = pdf.coeffs()[5]   
        lam0[9] = pdf.coeffs()[7]   

    
    return(lam0)
