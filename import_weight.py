import numpy as np
import pandas as pd
import scipy as sc 
 
# from import_dynamic import dynamicMeas
# from main import weightOld
import import_dynamic
import import_static
import mainDynamic


def findFuelMoments():
    '''
    DESCRIPTION:    Gives interpolant of the fuel moment as a function of the fuel mass. The excel file containing the fuel data needs to have the name 'FuelMoments'.
    ========
    INPUT:\n
    ... None'n

    OUTPUT:\n
    ... MomF [Type]         Interpolant of the fuel moment as a function of fuel mass
    '''

    dfm = pd.read_excel('FuelData/FuelMoments.xlsx',usecols='A')

    masslst = np.arange(100,5000,100)
    masslst = np.append(masslst,5008)
    
    #Convert to SI
    masslst = masslst*0.45359
    
    dfm = np.array(dfm)
    
    momentlst = np.append(298.16,dfm)
    
    #Convert to SI
    momentlst = momentlst*(0.45359*0.0254)*100
    
    MomF = sc.interpolate.interp1d(masslst,momentlst)
    
    return MomF
   

def calcWeightCG(inputFile, dataSet):
    '''
    DESCRIPTION:    Gives dataframe for Weight and Xcg
    ========
    INPUT:\n
    ... inputFile [String]:         Choose between 'reference' or 'actual'\n
    ... dataSet [String]:           Choose between 'static1', 'static2a', 'static2b' or 'dynamic'\n

    OUTPUT:\n
    ... MassBal [Dataframe]:        Pandas dataframe containing the weight and xcg
    '''
    
    pay = mainDynamic.weightOld(inputFile)
    MBlockFuel = pay.mblockfuel
    MBem = pay.MBem
    MomBem = pay.MomBem

    if dataSet == 'dynamic':
        MeasData = import_dynamic.dynamicMeas(inputFile,SI=True)
    elif dataSet in ['static1','static2a','static2b']:
        MeasData = import_static.staticMeas(inputFile, dataSet)
    else:
        raise ValueError("invalid input for 'dataSet'; choose between 'static1', 'static2a', 'static2b' or 'dynamic'")

    LocSwitch = pay.locSwitch

    Xcgpay = np.array([131,131,214,214,251,251,288,288,170])*0.0254
    
    #Calculate all payload masses and moments
    
    Mpaylst = np.array([pay.mpilot1, pay.mpilot2, pay.mobserver1L, pay.mobserver1R, pay.mobserver2L, pay.mobserver2R, pay.mobserver3L, pay.mobserver3R, pay.mcoordinator])
    
    MomentPay = Mpaylst*Xcgpay
    
    #If we add baggage
    #XcgBag = np.array(74,321,338)*0.0254
    
    Mpay = np.sum(Mpaylst)
    MomentPay = np.sum(MomentPay)
    
    Mfuel = np.array([])
    
    if dataSet == 'dynamic':
        DynFusedlh = np.array(MeasData['lh_engine_FU'])
        DynFusedrh = np.array(MeasData['rh_engine_FU'])    
        
        Fused =   DynFusedlh + DynFusedrh
    
    elif dataSet in ['static1','static2a','static2b']:
        Fused = np.array(MeasData['F. Used'])
    
    for i in Fused:
        MNewFuel = MBlockFuel - i
        Mfuel = np.append(Mfuel,MNewFuel) 
    
    #total mass of plane
    
    M = np.ones(len(Mfuel))*Mpay + np.ones(len(Mfuel))*MBem + Mfuel
    
    MomF = findFuelMoments()
    
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

    if dataSet == 'dynamic':
        MassBal['time'] = MeasData['time']

    MassBal = pd.DataFrame(data=MassBal)
    
    return MassBal





''' Delete this part once understood: to see how functions work '''
# MassBal1 = calcWeightCG('reference','static1')
# MassBal2a = calcWeightCG('reference','static2a')
# MassBal2b = calcWeightCG('reference','static2b')
# MassBalDyn = calcWeightCG('reference','dynamic')

# print(MassBal2a)