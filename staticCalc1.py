import numpy as np
import scipy.stats as stat
import pandas as pd
import matplotlib.pyplot as plt

import import_parameters as imPar
import import_static as imStat
import import_weight as imWeight


def calcAeroCoeff(inputFile, dataSet):
    '''
    DESCRIPTION:    This function calculates the lift coefficient for the static measurements. (CL \approx CN)
    ========
    INPUT:\n
    ... inputFile [String]:             Name of excel file; choose between 'reference' or 'actual'\n
    ... dataSet [String]:               Name of data set; choose between 'static1', 'static2a' or 'static2b'\n
    ... SI [Condition]:                 By default set to SI=True\n

    OUTPUT:\n
    ... aeroCoeff [Dataframe]:          Pandas dataframe containing Cl, Cd, e, Cla, Cd0, aoa0   
    '''

    # Import data

    param            = imPar.parametersStatic()
    static           = imStat.staticMeas(inputFile, dataSet)
    staticNotSI      = imStat.staticMeas(inputFile, dataSet, SI=False)
    staticFlightCond = imStat.staticFlightCondition(inputFile, dataSet)
    staticTp         = imStat.staticThrust(inputFile, dataSet)
    staticWeight     = imWeight.calcWeightCG(inputFile, dataSet)

    # Obtain vales from data
    S   = param.S
    A   = param.A
    aoa_rad = static['aoa'].to_numpy()
    aoa_deg = staticNotSI['aoa'].to_numpy()
    rho = staticFlightCond['rho'].to_numpy()
    Vt  = staticFlightCond['Vt'].to_numpy()
    W   = staticWeight['Weight'].to_numpy()
    Tp  = staticTp['Tp'].to_numpy()

    # Calculations
    CL = W / (0.5 * rho * Vt**2 * S)
    CD = Tp / (0.5 * rho * Vt**2 * S)
    
    aeroCoeff = {}

    if dataSet == 'static1':
        CL_aoa = stat.linregress(aoa_rad,CL)
        CD_CL2 = stat.linregress(CL**2,CD)

        CLa = CL_aoa.slope
        aoa0 = -CL_aoa.intercept / CLa
        e = 1/(np.pi*A*CD_CL2.slope)
        CD0 = CD_CL2.intercept

        dataNames = ['CL','CD']
        for name in dataNames:
            aeroCoeff[name] = locals()[name]  

    else:
        dataNames = ['CL','CD']
        for name in dataNames:
            aeroCoeff[name] = locals()[name]
    aeroCoeff = pd.DataFrame(data=aeroCoeff)

    return CLa, aoa0, e, CD0, aeroCoeff

def plotPolar(inputFile):
    '''
    DESCRIPTION:    This function plots the CL-alpha and CL-CD curve
    ========
    INPUT:\n
    ... inputFile [String]:             Name of excel file; choose between 'reference' or 'actual'\n


    OUTPUT:\n
    '''

    static1 = imStat.staticMeas(inputFile,'static1')
    static1NotSI = imStat.staticMeas(inputFile, 'static1', SI=False)
    aeroCoeff = calcAeroCoeff(inputFile,'static1')[4]

    aoa_rad = static1['aoa'].to_numpy()
    aoa_deg = static1NotSI['aoa'].to_numpy()
    CL = aeroCoeff['CL'].to_numpy()
    CD = aeroCoeff['CD'].to_numpy()
    return









''' Run these lines to test if all functions work properly without any coding errors '''

# inputFile = 'reference'

# aeroCeoff1 = calcAeroCoeff(inputFile, 'static1')
# aeroCoeff2a = calcAeroCoeff(inputFile, 'static2a')
# aeroCoeff2b = calcAeroCoeff(inputFile, 'static2b')

# print(aeroCeoff1)
# print(aeroCoeff2a)
# print(aeroCoeff2b)
