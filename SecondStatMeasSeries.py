
def calcElevEffectiveness(param, secMeasSeries_2):
    '''
    This function calculates the elevator effectiveness Cmdelta, which is a constant value found from
    the measurements when changing xcg.
    ----
    Input:  param [class]                           constant parameters
            secMeasSeries_2 [array]                 meassurements taken (serie 2.2)
    
    Output: Cmdelta [float]                         elevator effectiveness
    '''

    # Constant values
    cbar = param.c
    delta = secMeasSeries_2['delta']
    delta1 = delta[0]
    delta2 = delta[1]

    # Function values
    CN = 0#lift function ...
    xcg1 = 0#weight function ...
    xcg2 = 0#weight function ...

    Cmdelta = ( CN * (xcg2 - xcg1) / cbar ) / ( delta2 - delta1 )

    return Cmdelta

def calElevDeflection(param, secMeasSeries_1, secMeasSeries_2):
    '''
    This function calculates the reduced elevator deflection for each meassurement taken during the second
    stationary meassurement series.
    ----
    Input:  param [class]                           constant parameters
            secMeasSeries_1 [array]                 measurements taken (serie 2.1)
            secMeasSeries_2 [array]                 measurements taken (serie 2.2)

    Output: delta_Red [array]                       reduced elevator deflection
    '''

    # Constant values
    CmTc = param.CmTc
    delta = secMeasSeries_1['delta']

    # Function values
    Tcs = 0#thrust function ...
    Tc = 0#thrust function ...
    Cmdelta = calcElevEffectiveness(param, secMeasSeries_2)

    delta_Red = delta - CmTc * (Tcs - Tc) / Cmdelta

    return delta_Red

def calcElevContrForce(param, secMeasSeries_1):
    '''
    This function calculates the reduced elevator control force for each meassurement taken during the second
    stationary measurement series.
    ----
    Input:  param [class]                           constant parameters
            secMeasSeries_1 [array]                 measurements taken (serie 2.1)

    Output: Fe_Red [array]                          reduced elevator control force
    '''

    # Constant values
    Ws = param.Ws
    Fe = secMeasSeries_1['Fe']

    # Function values
    W = 0#weight function ...

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

