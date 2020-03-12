# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:51:59 2020

@author: afras
"""

from math import *
import numpy as np
import pandas as pd
import scipy.interpolate

def CalcWeightCG(Mbem,MomEmp,Mfuel,Fused,MassPass,Pos1,Pos2):
    
    Xcgpay = np.array(131,131,214,214,251,251,288,288,170)*0.0254
    Momentpay = MassCrew*Xcgpay
    
    XcgBag = np.array(74,321,338)*0.0254
    Mpayload = sum(Mpayload)
    
    Mfuel = [Mfuel]
    
    for i in t:
        Mfuel1 = Mfuel0 - Fused[0]
        Mfuel = np.append(Mfuel,Mfuel1)
    M = Mpayload + Mbem + Mfuel
    
    #Reading the list for fuel moments
    
    dfm = pd.read_excel(r'file:///C:/Users/afras/OneDrive/Documents/SVV_FD_sim/FuelMoments.xlsx',usecols='A')

    masslst = np.arange(100,5000,100)
    masslst = np.append(masslst,5008)
    
    #Convert to SI
    masslst = masslst**0.45359
    
    dfm = np.array(dfm)
    
    momentlst = np.append(298.16,dfm[:-1])
    
    #Convert to SI
    momentlst = momentlst*(0.45359*0.0254)
    
    
    MomF = scipy.interpolate.interp1d(masslst,momentlst)
    
    
    
    Xcg = (MomP + MomEmp + MomF(Mfuel))/M
    
    

    return M





