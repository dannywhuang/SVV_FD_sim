import numpy as np 

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
    delta1 = secMeasSeries_2#[index]
    delta2 = secMeasSeries_2#[index]

    # Function values
    CN = 0#function calculating lift coefficient
    xcg1 = 0#function calculating center of gravity location
    xcg2 = 0#function calculating center of gravity location

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
    delta = secMeasSeries_1#[index]

    # Function values
    Tcs = 0#function calculating the thrust coefficient
    Tc = 0#function calculating the thrust coefficient
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
    Fe = secMeasSeries_1#[index]

    # Function values
    W = 0#function calculating the weight at meassurement condition

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

