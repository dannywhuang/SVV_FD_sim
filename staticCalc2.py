import scipy as sc
import numpy as np

from main import ParametersOld
from import_static import staticMeas, staticThrust
from staticCalc1 import calcLiftCoeff


def calcElevEffectiveness(inputFile, SI=True):
    '''
    DESCRIPTION:    This function calculates the elevator effectiveness (Cmdelta), which is a constant value found from the measurements when changing xcg.
    ========
    INPUT:\n
    ... static2b [Dataframe]:       Panda dataframe containing the meassurements taken (serie 2b)\n

    OUTPUT:\n
    ... Cmdelta [float]:            Elevator effectiveness
    '''

    # Import data
    param = ParametersOld()
    static2b = staticMeas(inputFile, 'static2a', SI)

    # Obtain values from data
    cbar   = param.c
    delta  = static2b['delta'].to_numpy()
    delta1 = delta[0]
    delta2 = delta[1]
    CN     = np.average( calcLiftCoeff(inputFile, 'static2b', SI) )
    xcg1   = 123#weight function ...
    xcg2   = 123#weight function ...

    # Calculation
    Cmdelta = ( CN * (xcg2 - xcg1) / cbar ) / ( delta2 - delta1 )
    return Cmdelta


def calcElevDeflection(inputFile, SI=True):
    '''
    DESCRIPTION:    This function calculates the reduced elevator deflection for each meassurement taken during the second stationary meassurement series.
    ========
    INPUT:\n
    ... static2a [Dataframe]:       Panda dataframe containing the measurements taken (serie 2a)\n
    ... static2b [Dataframe]:       Panda dataframe containing the meassurements taken (serie 2b)\n

    OUTPUT:\n
    ... deltaRed [Array]:           Numpy array containing the reduced elevator deflection\n
    ... Cma [float]:                Longitudinal stability parameter
    
    '''

    # Import data
    param = ParametersOld()
    static2a = staticMeas(inputFile, 'static2a', SI)
    staticThrust2aRed = staticThrust(inputFile, 'static2a', standard=True)
    staticThrust2a = staticThrust(inputFile, 'static2a', standard=False)

    # Obtain values from data
    CmTc = param.CmTc
    delta = static2a['delta'].to_numpy()
    aoa = static2a['aoa'].to_numpy()
    Tcs = staticThrust2aRed['Tp'].to_numpy()
    Tc  = staticThrust2a['Tp'].to_numpy()
    Cmdelta = calcElevEffectiveness(inputFile, SI)

    # Calculations
    deltaRed = delta - CmTc * (Tcs - Tc) / Cmdelta
    linregress = sc.stats.linregress(aoa, delta)
    Cma = linregress.slope * - Cmdelta

    return deltaRed, Cma


def calcElevContrForce(inputFile, SI=True):
    '''
    DESCRIPTION:    This function calculates the reduced elevator control force for each meassurement taken during the second stationary measurement series.
    ========
    INPUT:\n
    ... param [class]:              Constant parameters\n
    ... static2a [Dataframe]:       Panda dataframe containing the meassurements taken (serie 2a)\n

    OUTPUT:\n
    ... Fe_Red [Arary]:             Numpy array containing the reduced elevator control force
    '''

    # Import data
    param = ParametersOld()
    static2a = staticMeas(inputFile, SI)

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
Cmdelta       = calcElevEffectiveness('reference', SI=True)
deltaRed, Cma = calcElevDeflection('reference', SI=True) 
FeRed         = calcElevContrForce('reference', SI=True)

print('\nCmdelta =',Cmdelta, '\n')
print('reduced elevator deflections:',deltaRed, ', longitudinal stability:', Cma, '\n')
print('reduced elevator control force:', FeRed, '\n')
