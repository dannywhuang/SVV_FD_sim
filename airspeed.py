
from numpy import sqrt


def calcV(param, meas):
    '''
    DESCRIPTION:    This function calculates both the equivalent airspeed and the reduced equivalent airspeed for every measurement.
    ========
    INPUT:\n
    ... param [Class]:              Constant parameters\n
    ... meas [Dataframe]:           Pandas dataframe containing measurement data\n

    OUTPUT:\n
    ... Ve [array]:                 Numpy array containing equivelent airspeed\n
    ... VeRed [array]:              Numpy array containing reduced equivalent airspeed
    '''

    # Constant values
    pres0 = param.pres0
    rho0  = param.rho0 
    Temp0 = param.Temp0 
    g0    = param.g 
    Ws    = param.Ws

    gamma = param.gamma
    lamb  = param.lamb 
    R     = param.R

    V        = meas['Vi'].to_numpy()

    TempMeas = meas['TAT'].to_numpy()
    hp       = meas['hp'].to_numpy()

    # Function values
    W = 123#weight function ...

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






''' Delete this part once understood: to see how functions work '''
from import_static import staticMeas
from import_dynamic import sliceTime
from main import ParametersOld

param = ParametersOld()
static1  = staticMeas('static1' , 'reference')
static2a = staticMeas('static2a', 'reference')
static2b = staticMeas('static2b', 'reference')

Ve, VeRed = calcV(param, static2a)

print('\nEquivalent airspeed:',Ve, '\nReduced equivalent airspeed:', VeRed, '\n')

