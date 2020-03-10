import pandas as pd 
from math import sqrt

def calcV(param, meas):
    '''
    This function calculates both the equivalent airspeed and the reduced equivalent airspeed for every measurement.
    ----
    Input:  param [class]                           constant parameters
            meas [array]                            meassurements taken (serie 2.2)
    
    Output: Ve [array]                              equivalent airspeed
            VeRed [array]                            reduced quivalent airspeed
    '''

    # Constant values
    pres0 = param.p0
    rho0  = param.rho0 
    Temp0 = param.Temp0 
    g0    = param.g 
    Ws    = param.Ws

    gamma = param.gamma
    lamb  = param.lamb 
    R     = param.R

    V        = meas['Vias']
    TempMeas = meas['TAT']
    hp       = meas['hp']

    # Function values
    W = 0#weight function ...

    # Calculation
    p = pres0 * (1 + lamb * hp / Temp0) ** (- g0 / (lamb * R))
    M = sqrt(2 / (gamma - 1) * ( (1 + pres0 / p * ( (1 + (gamma - 1)/(2 * gamma) * rho0 / pres0 * V**2)**( (gamma -1) / gamma ) -1 ))**( (gamma - 1) / gamma ) - 1) )
    Temp = TempMeas / ( 1 + (lamb - 1)/2 * M**2 )
    a = sqrt(gamma * R * Temp)
    Vt = M * a
    rho = p / (R * Temp)
    Ve = Vt * sqrt(rho / rho0)

    VeRed = Ve * sqrt( Ws / W )

    return Ve, VeRed
