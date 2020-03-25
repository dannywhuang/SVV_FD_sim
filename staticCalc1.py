import numpy as np
import scipy.stats as stat
import pandas as pd
import matplotlib.pyplot as plt

import import_parameters as imPar
import import_static as imStat
import import_weight as imWeight



# Calculates CL, CD for all static1 measurements & calculates CLa, aoa0, e, CD0m using static1
def calcAeroCoeff(inputFile):
    '''
    DESCRIPTION:    This function calculates the aerodynamic coefficient for the static measurements. (CL \approx CN)
    ========
    INPUT:\n
    ... inputFile [String]:             Name of excel file; choose between 'reference' or 'actual'\n

    OUTPUT:\n
    ... aeroCoeff [Dataframe]:          CLa, aoa0, e,CD0, Pandas dataframe containing CL, CD  
    '''

    # Import data

    param            = imPar.parametersStatic()
    static           = imStat.staticMeas(inputFile, 'static1')
    staticNotSI      = imStat.staticMeas(inputFile, 'static1', SI=False)
    staticFlightCond = imStat.staticFlightCondition(inputFile, 'static1')
    staticTp         = imStat.staticThrust(inputFile, 'static1')
    staticWeight     = imWeight.calcWeightCG(inputFile, 'static1')

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
    CL_aoa = stat.linregress(aoa_rad,CL)
    CD_CL2 = stat.linregress(CL**2,CD)

    CLa = CL_aoa.slope
    aoa0 = -CL_aoa.intercept / CLa
    e = 1/(np.pi*A*CD_CL2.slope)
    CD0 = CD_CL2.intercept

    aeroCoeff = {}
    dataNames = ['CL','CD']
    for name in dataNames:
        aeroCoeff[name] = locals()[name]
    aeroCoeff = pd.DataFrame(data=aeroCoeff)

    return CLa, aoa0, e, CD0, aeroCoeff


# Makes the lift curve using the static1 measurements
def plotLift(inputFile):
    '''
    DESCRIPTION:    This function plots the CL-alpha curve
    ========
    INPUT:\n
    ... inputFile [String]:             Name of excel file; choose between 'reference' or 'actual'\n


    OUTPUT:\n
    '''

    # Import the data
    param = imPar.parametersStatic()
    static1NotSI = imStat.staticMeas(inputFile, 'static1', SI=False)
    aeroCoeff = calcAeroCoeff(inputFile)[-1]
    static1FlightCond = imStat.staticFlightCondition(inputFile, 'static1')

    # Get the required parameters
    aoa_deg = static1NotSI['aoa'].to_numpy()
    CL = aeroCoeff['CL'].to_numpy()

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

    return


# Makes the drag polar using the static1 measurements
def plotPolar(inputFile):
    '''
    DESCRIPTION:    This function plots the CL-CD curve
    ========
    INPUT:\n
    ... inputFile [String]:             Name of excel file; choose between 'reference' or 'actual'\n


    OUTPUT:\n
    '''

    # Import the data
    param = imPar.parametersStatic()
    static1NotSI = imStat.staticMeas(inputFile, 'static1', SI=False)
    aeroCoeff = calcAeroCoeff(inputFile)[-1]
    static1FlightCond = imStat.staticFlightCondition(inputFile, 'static1')

    # Get the required parameters
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
