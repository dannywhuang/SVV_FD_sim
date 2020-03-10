import scipy as sc


def calcElevEffectiveness(param, static2b):
    '''
    This function calculates the elevator effectiveness Cmdelta, which is a constant value found from
    the measurements when changing xcg.
    ----
    Input:      param [class]                           constant parameters
                static2b [Dataframe]                    meassurements taken (serie 2b)
    
    Output:     Cmdelta [float]                         elevator effectiveness
    '''

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
    Cmdelta = -1.1642

    return Cmdelta


def calcElevDeflection(param, static2a, static2b):
    '''
    This function calculates the reduced elevator deflection for each meassurement taken during the second
    stationary meassurement series.
    ----
    Input:      param [class]                           constant parameters
                static2a [Dataframe]                    measurements taken (serie 2a)
                static2b [Dataframe]                    measurements taken (serie 2b)

    Output:     delta_Red [Dataframe]                   reduced elevator deflection
                Cma [float]                             ongitudinal stability
    '''

    # Constant values
    CmTc = param.CmTc
    delta = static2a['delta'].to_numpy()
    aoa = static2a['aoa'].to_numpy()

    # Function values
    Tcs = 123#thrust function ...
    Tc = 123#thrust function ...
    Cmdelta = calcElevEffectiveness(param, static2b)

    # Calculations
    delta_Red = delta - CmTc * (Tcs - Tc) / Cmdelta

    linregress = sc.stats.linregress(aoa, delta)
    Cma = linregress.slope * - Cmdelta

    return delta_Red, Cma


def calcElevContrForce(param, static2a):
    '''
    This function calculates the reduced elevator control force for each meassurement taken during the second
    stationary measurement series.
    ----
    Input:      param [class]                           constant parameters
                static2a [array]                        measurements taken (serie 2a)

    Output:     Fe_Red [Dataframe]                      reduced elevator control force
    '''

    # Constant values
    Ws = param.Ws
    Fe = static2a['Fe']

    # Function values
    W = 123#weight function ...

    # Calculation
    Fe_Red = Fe * Ws / W

    return Fe_Red


def plotElevTrimCurve():
    '''
    Short description of what the function does
    ----
    Input:  name [type]                             description of the variable
            name2 [type2]                           description of the variable2

    Output: name [type]                             description of the variable
            name2 [type2]                           description fothe varaible2
    '''


    return


def plotElevContrForceCurve():
    '''
    Short description of what the function does
    ----
    Input:    name [type]                             description of the variable
              name2 [type2]                           description of the variable2

    Output:   name [type]                             description of the variable
              name2 [type2]                           description fothe varaible2
    '''


    return 

