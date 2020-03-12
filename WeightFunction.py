# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:51:59 2020

@author: afras
"""

from math import *
import numpy as np
import pandas as pd
import scipy.interpolate
from import_static import static1, static2a, static2b
def FindFuelMoments(fileName):
    
    dfm = pd.read_excel(fileName,usecols='A')

    masslst = np.arange(100,5000,100)
    masslst = np.append(masslst,5008)
    
    #Convert to SI
    masslst = masslst**0.45359
    
    dfm = np.array(dfm)
    
    momentlst = np.append(298.16,dfm)
    
    #Convert to SI
    momentlst = momentlst*(0.45359*0.0254)
    
    
    MomF = scipy.interpolate.interp1d(masslst,momentlst)
    
    return MomF


filename = 'file:///C:/Users/afras/OneDrive/Documents/SVV_FD_sim/FuelMoments.xlsx'
    

def CalcWeightCG(filenameFM,MeasData,Mbem,MomEmp,MBlockFuel,MassPass,Pos1,Pos2):
    
    """
    MeasData is either static1, static2a or static2b
    """
    
    Xcgpay = np.array(131,131,214,214,251,251,288,288,170)*0.0254
    Momentpay = MassCrew*Xcgpay
    
    XcgBag = np.array(74,321,338)*0.0254
    Mpay = sum(Mpay)
    
    Mfuel = [MBlockFuel]
    
    Fused = np.array(MeasData['F. Used'])
    
    for i in Fused:
        MNewFuel = MBlockFuel - Fused[i]
        Mfuel = np.append(Mfuel,MNewFuel) 
    
    #total mass of plane
    
    M = np.ones(np.len(Mfuel))*Mpay + np.ones(np.len(Mfuel))*MBem + Mfuel
    
    
    MomF = FindFuelMoments(filenameFM)
    
    
    Xcg = (MomP + MomEmp + MomF(Mfuel))/M
    
    

    return M

Fused1 = np.array(static1['F. Used'])

print(Fused1)
"""



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
    
"""
