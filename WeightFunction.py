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
from main import weightOld


#These are constants idk where to put them so here they are
MBem = 9165*0.45359
MomBem = 2672953.5*0.45359*0.0254

#Initial Seat Location of the person who switched
LocSwitch = 7

"""
The dynamic data, uncomment or define somewhere if you wanna test on dynamic
"""
DynData = pd.read_csv('dynamicData/reference_SI.csv')


def FindFuelMoments(fileName):
    '''
    DESCRIPTION: gives interpolant of the fuel moment as a function of the fuel mass
    =========
    
    input = name of the FuelMoments excel file 
    '''
    
    dfm = pd.read_excel(fileName,usecols='A')

    masslst = np.arange(100,5000,100)
    masslst = np.append(masslst,5008)
    
    #Convert to SI
    masslst = masslst*0.45359
    
    dfm = np.array(dfm)
    
    momentlst = np.append(298.16,dfm)
    
    #Convert to SI
    momentlst = momentlst*(0.45359*0.0254)*100
    
    
    MomF = scipy.interpolate.interp1d(masslst,momentlst)
    
    return MomF


   

def CalcWeightCG(filenameFM,MeasData,Mbem,MomBem,weightOld,LocSwitch):
    
    '''
    DESCRIPTION: gives dataframe for Weight and Xcg
    =========
    
    input = name of the FuelMoments excel file filenameFM 
    
    MeasData is either static1, static2a, static2b or DynData
    
    LocSwitch is the seat location of the person who moved, which is not available from the excell sheet.
    
    It is only used for static2b, so place random number if you use other MeasData 
    '''
    
    
    pay = weightOld()
    
    MBlockFuel = pay.mblockfuel

    Xcgpay = np.array([131,131,214,214,251,251,288,288,170])*0.0254
    
    #Calculate all payload masses and moments
    
    Mpaylst = np.array([pay.mpilot1, pay.mpilot2, pay.mobserver1L, pay.mobserver1R, pay.mobserver2L, pay.mobserver2R, pay.mobserver3L, pay.mobserver3R, pay.mcoordinator])
    
    MomentPay = Mpaylst*Xcgpay
    
    #If we add baggage
    #XcgBag = np.array(74,321,338)*0.0254
    
    Mpay = np.sum(Mpaylst)
    MomentPay = np.sum(MomentPay)
    
    
    Mfuel = np.array([])
    
    
    if len(MeasData) > 10000:
        DynFusedlh = np.array(MeasData['lh_engine_FU'])
        DynFusedrh = np.array(MeasData['rh_engine_FU'])    
        
        Fused =   DynFusedlh+ DynFusedrh
    
    else:
        Fused = np.array(MeasData['F. Used'])
    
    for i in Fused:
        MNewFuel = MBlockFuel - i
        Mfuel = np.append(Mfuel,MNewFuel) 
    
    #total mass of plane
    
    M = np.ones(len(Mfuel))*Mpay + np.ones(len(Mfuel))*MBem + Mfuel
    
    
    MomF = FindFuelMoments(filenameFM)
    
    MomFlst = np.array([])
    for fuel in Mfuel:
        MomFlst = np.append(MomFlst,MomF(fuel))
        
     
    
    Xcg = (MomentPay*np.ones(len(MomFlst)) + MomBem*np.ones(len(MomFlst)) + MomFlst)/M
    
    if len(Fused) == 2:
        MomentPay1 = MomentPay
        MomentPay2 = MomentPay - Mpaylst[LocSwitch]*pay.position1*0.0254 + Mpaylst[LocSwitch]*pay.position2*0.0254
        MomentPay = np.array([MomentPay1,MomentPay2])
        Xcg = (MomentPay*np.ones(len(MomFlst)) + MomBem*np.ones(len(MomFlst)) + MomFlst)/M
    
    #Transform Mass to Weight
    
    Weight = M*9.81
    
    #Transform numpy arrays to Dataframe
    MassBal = {}
    dataNames = ['Weight','Xcg']
    for name in dataNames:
        MassBal[name] = locals()[name]
    MassBal = pd.DataFrame(data=MassBal)
    
    return MassBal

#Test function

#MassBal = CalcWeightCG('file:///C:/Users/afras/OneDrive/Documents/SVV_FD_sim/FuelMoments.xlsx',static2b,MBem,MomBem,weightOld,LocSwitch)
#print(MassBal)
