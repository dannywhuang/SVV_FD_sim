import scipy.stats as stats
import numpy as np
import pandas as pd

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
    Cl = W / (0.5 * rho * Vt**2 * S)
    Cd = Tp / (0.5 * rho * Vt**2 * S)
    
    aeroCoeff = {}

    if dataSet == 'static1':
        Cl_aoa = stats.linregress(aoa_deg,Cl)
        Cd_Cl2 = stats.linregress(Cl**2,Cd)

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


def calcElevEffectiveness(inputFile):
    '''
    DESCRIPTION:    This function calculates the elevator effectiveness (Cmdelta), which is a constant value found from the measurements when changing xcg.
    ========
    INPUT:\n
    ... inputFile [String]:             Name of excel file; choose between 'reference' or 'actual'\n
    ... SI [Condition]:                 By default set to SI=True\n

    OUTPUT:\n
    ... Cmdelta [float]:                Elevator effectiveness
    '''

    # Import data
    param          = imPar.parametersStatic()
    static2b       = imStat.staticMeas(inputFile, 'static2b', SI=True)
    aeroCoeff      = calcAeroCoeff(inputFile, 'static2b')
    static2bWeight = imWeight.calcWeightCG(inputFile, 'static2b')

    # Obtain values from data
    cbar   = param.c
    delta  = static2b['delta'].to_numpy()
    delta1 = delta[0]
    delta2 = delta[1]
    Cn     = aeroCoeff['Cl'].to_numpy()
    CnAvg  = np.average( Cn )
    xcg    = static2bWeight['Xcg'].to_numpy()
    xcg1   = xcg[0]
    xcg2   = xcg[1]

    # Calculation
    Cmdelta = -1 * ( CnAvg * (xcg2 - xcg1) / cbar ) / ( delta2 - delta1 )
    return Cmdelta


def calcElevDeflection(inputFile):
    '''
    DESCRIPTION:    This function calculates the reduced elevator deflection for each meassurement taken during the second stationary meassurement series.
    ========
    INPUT:\n
    ... inputFile [String]:             Name of excel file; choose between 'reference' or 'actual'\n
    ... SI [Condition]:                 By default set to SI=True\n

    OUTPUT:\n
    ... deltaRed [Array]:               Numpy array containing the reduced elevator deflections
    ... Cma [float]:                    Longitudinal stability parameter
    '''

    # Import data
    param             = imPar.parametersStatic()
    static2a          = imStat.staticMeas(inputFile, 'static2a', SI=True)
    staticThrust2aRed = imStat.staticThrust(inputFile, 'static2a', standard=True)
    staticThrust2a    = imStat.staticThrust(inputFile, 'static2a', standard=False)

    # Obtain values from data
    CmTc  = param.CmTc
    delta = static2a['delta'].to_numpy()
    aoa   = static2a['aoa'].to_numpy()
    Tcs   = staticThrust2aRed['Tcs'].to_numpy()
    Tc    = staticThrust2a['Tc'].to_numpy()
    Cmdelta = calcElevEffectiveness(inputFile)

    # Calculations
    deltaRed = delta - CmTc * (Tcs - Tc) / Cmdelta
    linregress = stats.linregress(aoa, delta)
    Cma = linregress.slope * - Cmdelta

    return deltaRed, Cma


def calcElevContrForce(inputFile):
    '''
    DESCRIPTION:    This function calculates the reduced elevator control force for each meassurement taken during the second stationary measurement series.
    ========
    INPUT:\n
    ... inputFile [String]:             Name of excel file; choose between 'reference' or 'actual'\n
    ... SI [Condition]:                 By default set to SI=True\n

    OUTPUT:\n
    ... FeRed [Arary]:                  Numpy array containing the reduced elevator control force
    '''

    # Import data
    param          = imPar.parametersStatic()
    static2a       = imStat.staticMeas(inputFile, 'static2a', SI=True)
    static2aWeight = imWeight.calcWeightCG(inputFile, 'static2a')

    # Obtain values from data
    Ws = param.Ws
    Fe = static2a['Fe'].to_numpy()
    W  = static2aWeight['Weight'].to_numpy()

    # Calculation
    FeRed = Fe * Ws / W
    return FeRed


def plotElevTrimCurve():
    '''
    DESCRIPTION:    Function description
    ========
    INPUT:\n
    ... param [Type]:               Parameter description\n
    ... param [Type]:               Parameter description\n

    OUTPUT:\n
    ... param [Type]:               Parameter description
    '''


    return


def plotElevContrForceCurve():
    '''
    DESCRIPTION:    Function description
    ========
    INPUT:\n
    ... param [Type]:               Parameter description\n
    ... param [Type]:               Parameter description\n

    OUTPUT:\n
    ... param [Type]:               Parameter description
    '''


    return 







''' Delete this part once understood: to see how functions work '''

# inputFile = 'reference'

# Cmdelta       = calcElevEffectiveness(inputFile)
# deltaRed, Cma = calcElevDeflection(inputFile) 
# FeRed         = calcElevContrForce(inputFile)

# print('\nCmdelta =',Cmdelta, '\n')
# print('reduced elevator deflections:',deltaRed, ', longitudinal stability:', Cma, '\n')
# print('reduced elevator control force:', FeRed, '\n')
