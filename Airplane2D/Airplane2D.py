#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:11:37 2020

@author: jain
"""
import numpy as np
import matplotlib.pyplot as plt
import time

## Import extra files
import Cal_moments_samples as MOM
import transform_domain as td
import Initial_lam as Initlam
import GLeg_pts as GL
import pdf_MaxEnt as pdf
# import pdf_MaxEnt_extd as pdfext
import MaxEntPdf as MEP
import plot_prop_param as plotprop
import MonteCarlo_propagation as MCp
import moment_propagation as Mp

# import StateCorrelation_MC_PME as plotMCpmeCorrelation
# import StateCorrelation_MC as plotMCCorrelation
import moment_transform as mT
import convert2indexNotation as C2IN


#############################################################################################################################################################################
######## Example Dubins problem
import Parameters_Airplane_gust_2Dsystem as Param
Parameters, InitialConditions, ControlParam, xdim, udim, Distribution = Param.parameters()
orderx, orderu, dt, tf, MC_n = Parameters
ax_0, bx_0, au_0, bu_0 = InitialConditions
au, bu = ControlParam


    
    
#############################################################################################################################
########### Monte Carlo Simulation

MC_propagation, MC_Propagation = MCp.MonteCarlo_propagation(dt, tf, MC_n, ax_0, bx_0, au, bu, xdim, udim)    

########### Expectation value using Index Notation
First_moment, First_moment_array, Second_moment_raw, Second_moment_central, Sigma3_array, Third_moment_raw, Third_moment_central, Fourth_moment_raw, Fourth_moment_central = Mp.moment_propagation(tf, dt, InitialConditions, ControlParam, Distribution, 'CUT', MC_n, xdim, udim, orderx , orderu)


###########################################################################################################################################################################
#%%% 

# Time = [0.5]
# Time = [ 0, 1, 1.5, 2, 2.5, 3 ,3.5, 4.5, 5, 5.5, 6,6.5, 7, 7.5,8,8.5,9,10] ## 0.5, 4, 9.5 are calculated using minimize function (BFGS)
Time = [ 0, 0.5,1, 1.5, 2, 2.5, 3 ,3.5, 4, 4.5, 5, 5.5, 6,6.5, 7, 7.5,8,8.5,9,9.5, 10]
# FIRSTmom = []
# FIRSTmomarr = np.copy(0*First_moment_array)
# SECONDmom = []
# SECONDmomraw = []
# THIRDmom = []
# THIRDmomraw = []
# FOURTHmomraw = []
# SECONDmomrawCUT =[]
# THIRDmomrawCUT =[]
# FOURTHmomrawCUT=[]
for ct in range(len(Time)):
    print('.')
    print('TimeStep = ',Time[ct])
    timestep=int(Time[ct]/dt)    
    X = MC_propagation[MC_n*timestep:MC_n*(timestep+1), 1:]
    
    ns = np.size(X, 1)
    Nsamp = len(X)
    w=1/MC_n 
    
    s=1
    bL = np.amin(X, axis=0)
    bU = np.amax(X, axis=0)
    y1, mu1 = MOM.Cal_moments_samples(X, w, 1,'central')
    y2, m2r = MOM.Cal_moments_samples(X, w, 2, 'raw')
    y2, m2 = MOM.Cal_moments_samples(X, w, 2, 'central')
    y3, m3 = MOM.Cal_moments_samples(X, w, 3, 'central')
    y3, m3r = MOM.Cal_moments_samples(X, w, 3, 'raw')
    y4, m4r = MOM.Cal_moments_samples(X, w, 4, 'raw')
    y4, m4 = MOM.Cal_moments_samples(X, w, 4, 'central')

    # FIRSTmomarr[ct,0] = timestep
    # FIRSTmomarr[ct,1:] = mu1
    # FIRSTmom.append(mu1)
    # SECONDmomraw.append(m2r.T)
    # SECONDmom.append(m2.T)
    # THIRDmomraw.append(m3r.T)
    # THIRDmom.append(m3)
    # FOURTHmomraw.append(m4r.T)
    
    Xt, dtt   = td.transform_domain(X, bL, bU, -s*np.ones([1,ns]), s*np.ones([1,ns]))
    # Xtmc, dtt = td.transform_domain(Xt, -s*np.ones([1,ns]), s*np.ones([1,ns]),  bL, bU)  ## getting X back
    
    [_, m1t]=MOM.Cal_moments_samples(Xt,w,1,'raw')
    [_, m2t]=MOM.Cal_moments_samples(Xt,w,2,'raw')
    [_, m3t]=MOM.Cal_moments_samples(Xt,w,3,'raw')
    [_, m4t]=MOM.Cal_moments_samples(Xt,w,4,'raw')
    
    E10 = First_moment[timestep]
    E20 = Second_moment_raw[timestep]
    E30 = Third_moment_raw[timestep]
    E40 = Fourth_moment_raw[timestep]
    
    M10, M20, M30, M40 = C2IN.convert2indexNotation(E10, E20, E30, E40)
    E11, E22, E33, E44 = mT.moment_transform(bU, bL, E10, E20, E30, E40 )
    ## Converting it back in "index notation" format
    M1, M2, M3, M4 = C2IN.convert2indexNotation(E11, E22, E33, E44)
    
    # SECONDmomrawCUT.append(M20)
    # THIRDmomrawCUT.append(M30)
    # FOURTHmomrawCUT.append(M40)    
    
    # print(mu1); print(M10)
    # print(m2r.T); print(M20)    
    # print(m3r.T); print(M30)
    # print(m4r.T); print(M40)
    
    
    
    MOMmethod = 'CUT'
    MOMmethod = 'MC'
    
    ### lam0 calculation
    if MOMmethod == 'CUT':
        M = np.append( np.append(1, M1), M2)
    if MOMmethod == 'MC':
        M = np.append( np.append(1, m1t), m2t)
    
    M= np.squeeze(M)
    
    lam00 = np.squeeze(Initlam.Initial_lam(y1, y2, M))
    
    xl = -s*np.ones([1,ns])
    xu =  s*np.ones([1,ns])
    methodd = 'GL'
    
    
    ##########################   2nd MOMS constraint ####################################################################################
    y = np.vstack( (np.vstack((np.zeros([1,ns]), y1 )) , y2) )
    
    tic = time.time() 
    Y2, lam22 = MEP.MaxEntPdf(y, M, xl, xu, lam00, methodd)
    lam2 =lam22.x
    toc = time.time()
    print( toc-tic, 'sec Elapsed for order-2')
    print('Moments computed by',MOMmethod ,'method')
    print('function value for 2nd order= ', lam22.fun) 
    
##%%    
    ############################   3rd MOMS constraint ####################################################################################
    
    y = np.vstack( (y , y3) )
    if MOMmethod == 'CUT':
        M = np.append(M, M3)
    if MOMmethod == 'MC':
        M = np.append(M, m3t)
    
    lam0 = np.append(lam00, np.zeros([ len(y) - len(lam2) ]))
    lam0 = np.append(lam2, np.zeros([ len(y) - len(lam2) ]))
    
    tic = time.time() 
    Y3, lam33 = MEP.MaxEntPdf(y, M, xl, xu, lam0, methodd)
    lam3=lam33.x
    toc = time.time()
    print( toc-tic, 'sec Elapsed for order-3')
    print('function value for 3rd order= ',lam33.fun)    
##%%    
    #############################   4th MOMS constraint ####################################################################################
    
    y = np.vstack( (y , y4) )
    if MOMmethod == 'CUT':
        M = np.append(M, M4)
    if MOMmethod == 'MC':
        M = np.append(M, m4t)    
    # lam0 = np.append(lam00, np.zeros([ len(y) - len(lam2) ]))
    lam0 = np.append(lam3, np.zeros([ len(y) - len(lam3) ]))
    
    tic = time.time() 
    Y4, lam44 = MEP.MaxEntPdf(y, M, xl, xu, lam0, methodd)
    lam4=lam44.x
    toc = time.time()
    print( toc-tic, 'sec Elapsed for order-4')
    print('function value for 4th order= ',lam44.fun)
    

    
    #############################  Plotting figures  ####################################################################################
    Xtmc = Xt

    s=1
    [xx,zz] = np.meshgrid(np.linspace(-1*s,1*s,100),np.linspace(-1*s,1*s,100));
    [xx_real, zz_real] = np.meshgrid(np.linspace(bL[0], bU[0], 100),np.linspace(bL[1], bU[1], 100))
    
    [XXX,WW] = GL.GLeg_pts( np.array([29, 29, 29]), -1*s , 1*s )
    
    XX = np.zeros(np.shape(XXX))
    W = np.zeros(np.shape(WW))
    for i in range( len(W)):
        XX[i, 0] = XXX[len(XXX)-i-1, 0]
        W[i]  = WW[len(WW)-i-1]
    

    
    pent2=np.zeros(np.shape(xx));
    pent3=np.zeros(np.shape(xx));
    pent4=np.zeros(np.shape(xx));
    # pent5=np.zeros(np.shape(xx));
    # pent6=np.zeros(np.shape(xx));
    
    for i in range( len(xx)):
        for j in range(len(xx)):
            for k in range(len(W)):
                x = np.array([xx[i,j],zz[i,j]])
                x= x.reshape([len(x) , 1])
                
                pent2[i,j]=  W[k]*pdf.pdf_MaxEnt( x, lam2.T, Y2 )
                pent3[i,j]=  W[k]*pdf.pdf_MaxEnt( x, lam3.T, Y3 )
                pent4[i,j]=  W[k]*pdf.pdf_MaxEnt( x, lam4.T, Y4 )

                # pent2[i,j]=pent2[i,j] + W[k]*pdfext.pdf_MaxEnt_extd( x, lam2.T, Y2,  xl, xu,   np.eye(len(lam2)))               
                # pent3[i,j]=pent3[i,j] + W[k]*pdfext.pdf_MaxEnt_extd( x, lam3.T, Y3,  xl, xu,   np.eye(len(lam3)))
                # pent4[i,j]=pent4[i,j] + W[k]*pdfext.pdf_MaxEnt_extd( x, lam4.T, Y4,  xl, xu,   np.eye(len(lam4)))

    
    if MOMmethod == 'CUT':
        fig = plt.figure(figsize=(20, 25))
        # plt.scatter(Xtmc[:,0],Xtmc[:,1], 20, color='cyan')
        plt.scatter(X[:,0],X[:,1], 20, color='cyan')
        vec= np.linspace(0, np.max(pent2), 50 )
        vec1 = np.linspace(vec[0], vec[1], 4)
        vec2 = np.hstack( ( vec1,vec[2:]) )
        S1 = plt.contour(xx_real, zz_real, pent2, vec2,  linewidths=5)
        plt = plotprop.plot_prop_param('X','Y', plt)
        plt.show()
        plt.pause(1)
        plt.savefig('RESULTSCUT/PME_CUT_XY_'+str( Time[ct]) +'sec_order2.png'  , bbox_inches='tight' )
        # plt.clabel(S1,inline=1)
    #    plt.savefig('RESULTS/RootResultsSqTerms/PME_XY_inline_'+str( Time[ct]) +'sec_order2.png'  , bbox_inches='tight' )
        plt.close()
    
        plt.figure(figsize=(20, 25))
        # plt.scatter(Xtmc[:,0],Xtmc[:,1], 20, color='cyan')
        plt.scatter(X[:,0],X[:,1], 20, color='cyan')
        vec= np.linspace(0, np.max(pent3), 50 )
        vec1 = np.linspace(vec[0], vec[1], 4)
        vec2 = np.hstack( ( vec1,vec[2:]) )
        S1 = plt.contour(xx_real, zz_real, pent3, vec2,  linewidths=5)
        plt = plotprop.plot_prop_param('X','Y', plt)
        plt.show()
        plt.pause(1)
        plt.savefig('RESULTSCUT/PME_CUT_XY_'+str( Time[ct]) +'sec_order3.png', bbox_inches='tight' )
    #    plt.clabel(S1,inline=1)
    #    plt.savefig('RESULTS/RootResultsSqTerms/PME_XY_inline_'+str( Time[ct]) +'sec_order3.png', bbox_inches='tight' )
        plt.close()
        
    
        plt.figure(figsize=(20, 25))
        plt.scatter(X[:,0],X[:,1], 20, color='cyan')
        vec= np.linspace(0, np.max(pent4), 50 )
        vec1 =  np.linspace(vec[0], vec[1], 4)
        vec2 = np.hstack( ( vec1,vec[2:]) )
        S1 = plt.contour(xx_real, zz_real, pent4, vec2 ,  linewidths=5)
        plt = plotprop.plot_prop_param('X','Y', plt)
        plt.show()
        plt.pause(1)
        plt.savefig('RESULTSCUT/PME_CUT_XY_'+str( Time[ct]) +'sec_order4.png', bbox_inches='tight' )
    #    plt.clabel(S1,inline=1)
    #    plt.savefig('RESULTS/RootResultsSqTerms/PME_XY_inline_'+str( Time[ct]) +'sec_order4.png', bbox_inches='tight' )
        plt.close()

    if MOMmethod == 'MC':
        fig = plt.figure(figsize=(20, 25))
        # plt.scatter(Xtmc[:,0],Xtmc[:,1], 20, color='cyan')
        plt.scatter(X[:,0],X[:,1], 20, color='cyan')
        vec= np.linspace(0, np.max(pent2), 50 )
        vec1 = np.linspace(vec[0], vec[1], 4)
        vec2 = np.hstack( ( vec1,vec[2:]) )
        S1 = plt.contour(xx_real, zz_real, pent2, vec2,  linewidths=5)
        plt = plotprop.plot_prop_param('X','Y', plt)
        plt.show()
        plt.pause(1)
        plt.savefig('RESULTSMC/PME_MC_XY_'+str( Time[ct]) +'sec_order2.png'  , bbox_inches='tight' )
        # plt.clabel(S1,inline=1)
    #    plt.savefig('RESULTS/RootResultsSqTerms/PME_XY_inline_'+str( Time[ct]) +'sec_order2.png'  , bbox_inches='tight' )
        plt.close()
    
        plt.figure(figsize=(20, 25))
        # plt.scatter(Xtmc[:,0],Xtmc[:,1], 20, color='cyan')
        plt.scatter(X[:,0],X[:,1], 20, color='cyan')
        vec= np.linspace(0, np.max(pent3), 50 )
        vec1 = np.linspace(vec[0], vec[1], 4)
        vec2 = np.hstack( ( vec1,vec[2:]) )
        S1 = plt.contour(xx_real, zz_real, pent3, vec2,  linewidths=5)
        plt = plotprop.plot_prop_param('X','Y', plt)
        plt.show()
        plt.pause(1)
        plt.savefig('RESULTSMC/PME_MC_XY_'+str( Time[ct]) +'sec_order3.png', bbox_inches='tight' )
    #    plt.clabel(S1,inline=1)
    #    plt.savefig('RESULTS/RootResultsSqTerms/PME_XY_inline_'+str( Time[ct]) +'sec_order3.png', bbox_inches='tight' )
        plt.close()
        
    
        plt.figure(figsize=(20, 25))
        plt.scatter(X[:,0],X[:,1], 20, color='cyan')
        vec= np.linspace(0, np.max(pent4), 50 )
        vec1 =  np.linspace(vec[0], vec[1], 4)
        vec2 = np.hstack( ( vec1,vec[2:]) )
        S1 = plt.contour(xx_real, zz_real, pent4, vec2 ,  linewidths=5)
        plt = plotprop.plot_prop_param('X','Y', plt)
        plt.show()
        plt.pause(1)
        plt.savefig('RESULTSMC/PME_MC_XY_'+str( Time[ct]) +'sec_order4.png', bbox_inches='tight' )
    #    plt.clabel(S1,inline=1)
    #    plt.savefig('RESULTS/RootResultsSqTerms/PME_XY_inline_'+str( Time[ct]) +'sec_order4.png', bbox_inches='tight' )
        plt.close()
    
    



