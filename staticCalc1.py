import numpy as np

from main import ParametersOld
from import_static import staticFlightCondition


def calcLiftCoeff(inputFile, dataSet, SI=True):
    '''
    DESCRIPTION:    This function calculates the lift coefficient for the static measurements. (CL \approx CN)
    ========
    INPUT:\n
    ... inputFile [String]:             Name of excel file; choose between 'reference' or 'actual'\n
    ... dataSet [String]:               Name of data set; choose between 'static1', 'static2a' or 'static2b'\n
    ... SI [Condition]:                 By default set to SI=True\n

    OUTPUT:\n
    ... CL [Array]:                     Numpy array containing lift coefficients (CL \approx CN)
    '''

    staticFlightCond = staticFlightCondition(inputFile, dataSet, SI)
    param = ParametersOld()

    S   = param.S

    rho = staticFlightCond['rho'].to_numpy()
    Vt  = staticFlightCond['Vt'].to_numpy()
    W   = 123#function for the weight

    CL = W / (0.5 * rho * Vt**2 * S)
    return CL








''' Delete this part once understood: to see how the functions work '''
# CL1  = calcLiftCoeff('reference', 'static1')
# CL2a = calcLiftCoeff('reference', 'static2a')
# CL2b = calcLiftCoeff('reference', 'static2b')

# print(CL1, CL2a, CL2b)