#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 11:27:55 2019

@author: axj307
"""




def plot_prop_param(Xlabel, Ylabel, plt):
    
#    plt.rcParams['font.serif'] = 'Ubuntu'
#    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12    
    
    plt.xlabel( Xlabel, fontsize=40)
    plt.ylabel( Ylabel,  fontsize=40)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    
    return(plt)