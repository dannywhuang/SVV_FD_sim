import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt

# from import_static import staticMeas, staticFlightCondition, staticThrust
# from import_weight import calcWeightCG
# from main import ParametersStatic

import import_static
import import_weight 
import mainDynamic


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
    param            = mainDynamic.ParametersStatic()
    static           = import_static.staticMeas(inputFile, dataSet)
    staticNotSI      = import_static.staticMeas(inputFile, dataSet, SI=False)
    staticFlightCond = import_static.staticFlightCondition(inputFile, dataSet)
    staticTp         = import_static.staticThrust(inputFile, dataSet)
    staticWeight     = import_weight.calcWeightCG(inputFile, dataSet)

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
    Cl = W / (0.5 * rho * Vt**2 * S)
    Cd = Tp / (0.5 * rho * Vt**2 * S)
    
    aeroCoeff = {}

    if dataSet == 'static1':
        Cl_aoa = sc.stats.linregress(aoa_deg,Cl)
        Cd_Cl2 = sc.stats.linregress(Cl**2,Cd)

        Cla = Cl_aoa.slope
        aoa0 = -Cl_aoa.intercept / Cla
        e = 1/(np.pi*A*Cd_Cl2.slope)
        Cd0 = Cd_Cl2.intercept

        dataNames = ['Cl','Cd','e','Cla','Cd0','aoa0']
        for name in dataNames:
            aeroCoeff[name] = locals()[name]
    else:
        dataNames = ['Cl','Cd']
        for name in dataNames:
            aeroCoeff[name] = locals()[name]
    aeroCoeff = pd.DataFrame(data=aeroCoeff)

    return aeroCoeff

def plotPolar(inputFile):
    '''
    DESCRIPTION:    This function plots the Cl-alpha and Cl-Cd curve
    ========
    INPUT:\n
    ... inputFile [String]:             Name of excel file; choose between 'reference' or 'actual'\n


    OUTPUT:\n
    '''

    static1 = import_static.staticMeas(inputFile,'static1')
    aeroCoeff = calcAeroCoeff(inputFile,'static1')

    aoa = static1['aoa'].to_numpy()
    Cl = aeroCoeff['Cl'].to_numpy()
    Cd = aeroCoeff['Cd'].to_numpy()
    return


''' Delete this part once understood: to see how the functions work '''
# Cl1  = calcAeroCoeff('reference', 'static1')['Cl'].to_numpy()
# Cl2a = calcAeroCoeff('reference', 'static2a')['Cl'].to_numpy()
# Cl2b = calcAeroCoeff('reference', 'static2b')['Cl'].to_numpy()
# Cd1  = calcAeroCoeff('reference', 'static1')['Cd'].to_numpy()
# Cd2a = calcAeroCoeff('reference', 'static2a')['Cd'].to_numpy()
# Cd2b = calcAeroCoeff('reference', 'static2b')['Cd'].to_numpy()

# print(Cl1)
# print(Cl2a)
# print(Cl2b)
# print(Cd1)
# print(Cd2a)
# print(Cd2b)