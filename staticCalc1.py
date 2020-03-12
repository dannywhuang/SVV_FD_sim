import numpy as np
import scipy as sc
import pandas as pd

from main import ParametersOld
from import_static import staticMeas, staticFlightCondition, staticThrust


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
    param            = ParametersOld()
    static1          = staticMeas(inputFile, 'static1')
    staticFlightCond = staticFlightCondition(inputFile, dataSet)
    staticTp         = staticThrust(inputFile, dataSet)

    # Obtain vales from data
    S   = param.S
    A   = param.A
    aoa = static1['aoa'].to_numpy()
    rho = staticFlightCond['rho'].to_numpy()
    Vt  = staticFlightCond['Vt'].to_numpy()
    W   = 123#function for the weight
    Tp  = staticTp['Tp'].to_numpy()

    # Calculations
    Cl = W / (0.5 * rho * Vt**2 * S)
    Cd = Tp / (0.5 * rho * Vt**2 * S)

    Cl_aoa = sc.stats.linregress(aoa,Cl)
    Cd_Cl2 = sc.stats.linregress(Cl**2,Cd)

    Cla = Cl_aoa.slope
    aoa0 = -Cl_aoa.intercept / Cla
    e = 1/(np.pi*A*Cd_Cl2.slope)
    Cd0 = Cd_Cl2.intercept

    aeroCoeff = {}
    dataNames = ['Cl','Cd','e','Cla','Cd0','aoa0']
    for name in dataNames:
        aeroCoeff[name] = locals()[name]
    aeroCoeff = pd.DataFrame(data=aeroCoeff)

    return aeroCoeff






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