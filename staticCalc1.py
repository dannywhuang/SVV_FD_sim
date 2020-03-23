import numpy as np
import scipy.stats as stat
import pandas as pd
import matplotlib.pyplot as plt

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
    CL = W / (0.5 * rho * Vt**2 * S)
    CD = Tp / (0.5 * rho * Vt**2 * S)
    
    aeroCoeff = {}

    if dataSet == 'static1':
        CL_aoa = stat.linregress(aoa_rad,CL)
        CD_CL2 = stat.linregress(CL**2,CD)

        CLa = CL_aoa.slope
        aoa0 = -CL_aoa.intercept / CLa
        e = 1/(np.pi*A*CD_CL2.slope)
        CD0 = CD_CL2.intercept

        dataNames = ['CL','CD']
        for name in dataNames:
            aeroCoeff[name] = locals()[name]  

    else:
        dataNames = ['CL','CD']
        for name in dataNames:
            aeroCoeff[name] = locals()[name]
    aeroCoeff = pd.DataFrame(data=aeroCoeff)

    return CLa, aoa0, e, CD0, aeroCoeff

def plotPolar(inputFile):
    '''
    DESCRIPTION:    This function plots the CL-alpha and CL-CD curve
    ========
    INPUT:\n
    ... inputFile [String]:             Name of excel file; choose between 'reference' or 'actual'\n


    OUTPUT:\n
    '''

    # Import the data
    param = imPar.parametersStatic()
    static1NotSI = imStat.staticMeas(inputFile, 'static1', SI=False)
    aeroCoeff = calcAeroCoeff(inputFile,'static1')[-1]
    static1FlightCond = imStat.staticFlightCondition(inputFile, 'static1')

    # Get the required parameters
    aoa_deg = static1NotSI['aoa'].to_numpy()
    CL = aeroCoeff['CL'].to_numpy()
    CD = aeroCoeff['CD'].to_numpy()

    # Mach range
    M = static1FlightCond['Mach'].to_numpy()
    Mmax = np.max(M)
    Mmin = np.min(M)

    # Reynolds number range
    rho = static1FlightCond['rho'].to_numpy()
    Vt = static1FlightCond['Vt'].to_numpy()
    mac = param.c
    T = static1FlightCond['Temp'].to_numpy()*(9/5) # [Rankine]
    T0 = 518.7    # [Rankine]
    mu0 = 3.62E-7 # [lb s/ft2]

    mu = mu0 * (T/T0)**1.5 * ((T0 + 198.72)/(T + 198.72))
    mu = mu*(0.45359237/(0.3048**2))

    Re = rho*Vt*mac/mu
    Remax = np.max(Re)
    Remin = np.min(Re)

    # Plot lift curve
    plt.figure('Lift curve',[10,7])
    plt.title('Lift Curve',fontsize=22)
    plt.plot(aoa_deg,CL,marker='o')
    plt.xlabel(r'$\alpha$ [$\degree$]',fontsize=16)
    plt.ylabel(r'$C_L$ [$-$]',fontsize=16)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(1.03*aoa_deg[1],1.05*CL[-2],'Aircraft configuration: Clean'+'\nMach number range: '+str(round(Mmin,2))+' - '+str(round(Mmax,2))+'\nReynolds number range: '+'{:.2e}'.format(Remin)+' - '+'{:.2e}'.format(Remax),bbox=props,fontsize=16)
    plt.grid()

    # Plot drag polar
    plt.figure('Drag polar',[10,7])
    plt.title('Drag polar',fontsize=22)
    plt.plot(CD,CL,marker='o')
    plt.xlabel(r'$C_D$ [$-$]',fontsize=16)
    plt.ylabel(r'$C_L$ [$-$]',fontsize=16)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(1.03*CD[-3],1.05*CL[1],'Aircraft configuration: Clean'+'\nMach number range: '+str(round(Mmin,2))+' - '+str(round(Mmax,2))+'\nReynolds number range: '+'{:.2e}'.format(Remin)+' - '+'{:.2e}'.format(Remax),bbox=props,fontsize=16)
    plt.grid()

    return





''' Run these lines to test if all functions work properly without any coding errors '''

# inputFile = 'reference'

# aeroCeoff1 = calcAeroCoeff(inputFile, 'static1')
# aeroCoeff2a = calcAeroCoeff(inputFile, 'static2a')
# aeroCoeff2b = calcAeroCoeff(inputFile, 'static2b')

# print(aeroCeoff1)
# print(aeroCoeff2a)
# print(aeroCoeff2b)
