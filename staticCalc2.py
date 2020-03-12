import scipy as sc
import numpy as np

from main import ParametersOld
from import_static import staticMeas, staticThrust
from staticCalc1 import calcAeroCoeff


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
    param     = ParametersOld()
    static2b  = staticMeas(inputFile, 'static2b', SI=True)
    aeroCoeff = calcAeroCoeff(inputFile, 'static2b')

    # Obtain values from data
    cbar   = param.c
    delta  = static2b['delta'].to_numpy()
    delta1 = delta[0]
    delta2 = delta[1]
    Cn     = aeroCoeff['Cl'].to_numpy()
    CnAvg  = np.average( Cn )
    xcg1   = 123#weight function ...
    xcg2   = 122#weight function ...

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
    param = ParametersOld()
    static2a = staticMeas(inputFile, 'static2a', SI=True)
    staticThrust2aRed = staticThrust(inputFile, 'static2a', standard=True)
    staticThrust2a = staticThrust(inputFile, 'static2a', standard=False)

    # Obtain values from data
    CmTc  = param.CmTc
    delta = static2a['delta'].to_numpy()
    aoa   = static2a['aoa'].to_numpy()
    Tcs   = staticThrust2aRed['Tcs'].to_numpy()
    Tc    = staticThrust2a['Tc'].to_numpy()
    Cmdelta = calcElevEffectiveness(inputFile)

    # Calculations
    deltaRed = delta - CmTc * (Tcs - Tc) / Cmdelta
    linregress = sc.stats.linregress(aoa, delta)
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
    param = ParametersOld()
    static2a = staticMeas(inputFile, 'static2a',SI=True)
    

    # Obtain values from data
    Ws = param.Ws
    Fe = static2a['Fe'].to_numpy()
    W = 123#weight function ...

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
Cmdelta       = calcElevEffectiveness('reference')
deltaRed, Cma = calcElevDeflection('reference') 
FeRed         = calcElevContrForce('reference')

print('\nCmdelta =',Cmdelta, '\n')
print('reduced elevator deflections:',deltaRed, ', longitudinal stability:', Cma, '\n')
print('reduced elevator control force:', FeRed, '\n')
