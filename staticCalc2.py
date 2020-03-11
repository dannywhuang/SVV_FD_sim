import scipy as sc

from main import ParametersOld


def calcElevEffectiveness(static2b):
    '''
    DESCRIPTION:    This function calculates the elevator effectiveness (Cmdelta), which is a constant value found from the measurements when changing xcg.
    ========
    INPUT:\n
    ... static2b [Dataframe]:       Panda dataframe containing the meassurements taken (serie 2b)\n

    OUTPUT:\n
    ... Cmdelta [float]:            Elevator effectiveness
    '''

    param = ParametersOld()

    # Constant values
    cbar = param.c
    delta = static2b['delta'].to_numpy()
    delta1 = delta[0]
    delta2 = delta[1]

    # Function values
    CN = 123#lift function ...
    xcg1 = 123#weight function ...
    xcg2 = 123#weight function ...

    # Calculation
    Cmdelta = ( CN * (xcg2 - xcg1) / cbar ) / ( delta2 - delta1 )

    return Cmdelta


def calcElevDeflection(static2a, static2b):
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

    param = ParametersOld()

    # Constant values
    CmTc = param.CmTc
    delta = static2a['delta'].to_numpy()
    aoa = static2a['aoa'].to_numpy()

    # Function values
    Tcs = 123#thrust function ...
    Tc = 123#thrust function ...
    Cmdelta = calcElevEffectiveness(static2b)

    # Calculations
    deltaRed = delta - CmTc * (Tcs - Tc) / Cmdelta

    linregress = sc.stats.linregress(aoa, delta)
    Cma = linregress.slope * - Cmdelta

    return deltaRed, Cma


def calcElevContrForce(static2a):
    '''
    DESCRIPTION:    This function calculates the reduced elevator control force for each meassurement taken during the second stationary measurement series.
    ========
    INPUT:\n
    ... param [class]:              Constant parameters\n
    ... static2a [Dataframe]:       Panda dataframe containing the meassurements taken (serie 2a)\n

    OUTPUT:\n
    ... Fe_Red [Arary]:             Numpy array containing the reduced elevator control force
    '''

    param = ParametersOld()

    # Constant values
    Ws = param.Ws
    Fe = static2a['Fe'].to_numpy()

    # Function values
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
# from import_static import staticMeas

# param = ParametersOld()
# static2a = staticMeas('static2a', 'reference')
# static2b = staticMeas('static2b', 'reference')

# Cmdelta       = calcElevEffectiveness(static2b)
# deltaRed, Cma = calcElevDeflection(static2a, static2b) 
# FeRed         = calcElevContrForce(static2a)

# print('\nCmdelta =',Cmdelta, '\n')
# print('reduced elevator deflections:',deltaRed, ', longitudinal stability:', Cma, '\n')
# print('reduced elevator control force:', FeRed, '\n')
