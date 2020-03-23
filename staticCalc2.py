import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import import_parameters as imPar
import import_static as imStat
import import_weight as imWeight 


# Calculates CL and CD for all static measurements
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
    Cl = W / (0.5 * rho * Vt**2 * S)
    Cd = Tp / (0.5 * rho * Vt**2 * S)
    
    aeroCoeff = {}

    if dataSet == 'static1':
        Cl_aoa = stats.linregress(aoa_deg,Cl)
        Cd_Cl2 = stats.linregress(Cl**2,Cd)

        Cla = Cl_aoa.slope
        aoa0 = -Cl_aoa.intercept / Cla
        e = 1/(np.pi*A*Cd_Cl2.slope)
        Cd0 = Cd_Cl2.intercept

        dataNames = ['Cl','Cd','e','Cla','Cd0','aoa0']
        for name in dataNames:
            aeroCoeff[name] = locals()[name]
    else:
        dataNames = ['Cl','Cd']
        for name in dataNames:
            aeroCoeff[name] = locals()[name]
    aeroCoeff = pd.DataFrame(data=aeroCoeff)

    return aeroCoeff


# Calculates Cmdelta using the static measurement static2b
def calcElevEffectiveness(inputFile):
    '''
    DESCRIPTION:    This function calculates the elevator effectiveness (Cmdelta), which is a constant value found from the measurements when changing xcg.
    ========
    INPUT:\n
    ... inputFile [String]:             Name of excel file; choose between 'reference' or 'actual'\n
    ... SI [Condition]:                 By default set to SI=True\n

    OUTPUT:\n
    ... Cmdelta [float]:                Elevator effectiveness
    '''

    # Import data
    param          = imPar.parametersStatic()
    static2b       = imStat.staticMeas(inputFile, 'static2b', SI=True)
    aeroCoeff      = calcAeroCoeff(inputFile, 'static2b')
    static2bWeight = imWeight.calcWeightCG(inputFile, 'static2b')

    # Obtain values from data
    cbar   = param.c
    delta  = static2b['delta'].to_numpy()
    delta1 = delta[0]
    delta2 = delta[1]
    Cn     = aeroCoeff['Cl'].to_numpy()
    CnAvg  = np.average( Cn )
    xcg    = static2bWeight['Xcg'].to_numpy()
    xcg1   = xcg[0]
    xcg2   = xcg[1]

    # Calculation
    Cmdelta = -1 * ( CnAvg * (xcg2 - xcg1) / cbar ) / ( delta2 - delta1 )
    return Cmdelta


# Calculates the reduced elevator deflections for all static2a measurements and Cma using static2a measurements
def calcElevDeflection(inputFile):
    '''
    DESCRIPTION:    This function calculates the reduced elevator deflection for each meassurement taken during the second stationary meassurement series.
    ========
    INPUT:\n
    ... inputFile [String]:             Name of excel file; choose between 'reference' or 'actual'\n
    ... SI [Condition]:                 By default set to SI=True\n

    OUTPUT:\n
    ... deltaRed [Array]:               Numpy array containing the reduced elevator deflections
    ... Cma [float]:                    Longitudinal stability parameter
    '''

    # Import data
    param             = imPar.parametersStatic()
    static2a          = imStat.staticMeas(inputFile, 'static2a', SI=True)
    staticThrust2aRed = imStat.staticThrust(inputFile, 'static2a', standard=True)
    staticThrust2a    = imStat.staticThrust(inputFile, 'static2a', standard=False)

    # Obtain values from data
    CmTc  = param.CmTc
    delta = static2a['delta'].to_numpy()
    aoa   = static2a['aoa'].to_numpy()
    Tcs   = staticThrust2aRed['Tcs'].to_numpy()
    Tc    = staticThrust2a['Tc'].to_numpy()
    Cmdelta = calcElevEffectiveness(inputFile)

    # Calculations
    deltaRed = delta - CmTc * (Tcs - Tc) / Cmdelta
    linregress = stats.linregress(aoa, delta)
    Cma = -1* linregress.slope * Cmdelta

    # #--
    # aoa = np.rad2deg(aoa)
    # linregress = stats.linregress(aoa, delta)

    # plt.figure('Elevator trim curve',[10,7])
    # plt.scatter(aoa,delta)
    # plt.plot(np.sort(aoa),np.sort(aoa)*linregress.slope + linregress.intercept, 'k--')
    # plt.ylim(1.2*max(delta),1.2*min(delta))
    # plt.grid()

    # plt.title("Elevator Trim Curve",fontsize=22)

    # plt.xlim(0.9*np.sort(aoa)[0],1.1*np.sort(aoa)[-1])
    # plt.xlabel('aoa   ($\degree$)',fontsize=16)
    # plt.ylabel('$\delta_{e}$   ($\degree$)',fontsize=16)
    # plt.axhline(0,color='k')

    # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    # plt.text(1.03*np.sort(aoa)[1],1.05*delta[1],'$x_{cg}$ = 7.15 m\nW = 59875 kg',bbox=props,fontsize=16)

    # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    # plt.show()

    return deltaRed, Cma


# Calculates the reduced elevator control force for all static2a measurements
def calcElevContrForce(inputFile):
    '''
    DESCRIPTION:    This function calculates the reduced elevator control force for each meassurement taken during the second stationary measurement series.
    ========
    INPUT:\n
    ... inputFile [String]:             Name of excel file; choose between 'reference' or 'actual'\n
    ... SI [Condition]:                 By default set to SI=True\n

    OUTPUT:\n
    ... FeRed [Arary]:                  Numpy array containing the reduced elevator control force
    '''

    # Import data
    param          = imPar.parametersStatic()
    static2a       = imStat.staticMeas(inputFile, 'static2a', SI=True)
    static2aWeight = imWeight.calcWeightCG(inputFile, 'static2a')

    # Obtain values from data
    Ws = param.Ws
    Fe = static2a['Fe'].to_numpy()
    W  = static2aWeight['Weight'].to_numpy()

    # Calculation
    FeRed = Fe * Ws / W
    return FeRed


# Makes the reduced elevator trim curve using static2a measurements
def plotElevTrimCurve(inputFile):
    '''
    DESCRIPTION:    Function description
    ========
    INPUT:\n
    ... inputFile [String]:             Name of excel file; choose between 'reference' or 'actual'\n

    OUTPUT:\n
    ... None, but running this function creates a figure which can be displayed by calling plt.plot()
    '''

    # Import data
    flightCond2b = imStat.staticFlightCondition(inputFile,'static2a')
    Weight2a     = imWeight.calcWeightCG(inputFile, 'static2a')
    
    # Obtain values from data
    VeRed    = np.sort(flightCond2b['VeRed'].to_numpy())
    deltaRed = np.rad2deg( np.sort( calcElevDeflection(inputFile)[0] ) )
    Xcg      = np.average(Weight2a['Xcg'].to_numpy())
    Weight   = np.average(Weight2a['Weight'].to_numpy())

    # Start plotting
    plt.figure('Elevator Trim Curve',[10,7])
    plt.title("Reduced Elevator Trim Curve",fontsize=22)
    plt.plot(VeRed,deltaRed,marker='o')

    plt.xlim(0.9*VeRed[0],1.1*VeRed[-1])
    plt.xlabel(r'$V_{e}^{*}$   ($\dfrac{m}{s}$)',fontsize=16)
    plt.ylim(1.2*deltaRed[-1],1.2*deltaRed[0])
    plt.ylabel(r'$\delta_{e}^{*}$   ($\degree$)',fontsize=16)
    plt.axhline(0,color='k')

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(1.03*VeRed[1],1.05*deltaRed[1],'$x_{cg}$ = '+str(round(Xcg,2))+' m\nW = '+str(int(round(Weight,0)))+' kg',bbox=props,fontsize=16)
    plt.grid()
    return


# Makes the reduced elevator control force curve using static2a measurements
def plotElevContrForceCurve(inputFile):
    '''
    DESCRIPTION:    Function description
    ========
    INPUT:\n
    ... inputFile [String]:             Name of excel file; choose between 'reference' or 'actual'\n

    OUTPUT:\n
    ... None, but running this function creates a figure which can be displayed by calling plt.plot()
    '''


    # Import data
    param        = imPar.parametersStatic()
    static2a     = imStat.staticMeas(inputFile,'static2a')
    flightCond2a = imStat.staticFlightCondition(inputFile,'static2a')
    Weight2a     = imWeight.calcWeightCG(inputFile, 'static2a')
    
    # Obtain values from data
    Ws    = param.Ws
    VeRed = np.sort( flightCond2a['VeRed'].to_numpy() )
    Fe    = static2a['Fe'].to_numpy()
    W     = Weight2a['Weight'].to_numpy()
    Xcg   = np.average(Weight2a['Xcg'].to_numpy())
    deltaTr = np.rad2deg(np.average(static2a['deltaTr']))

    # Calculation
    FeRed = np.sort( Fe*Ws/W )

    # Start plotting
    plt.figure('Elevator Force Control Curve',[10,7])
    plt.title("Reduced Elevator Control Force Curve",fontsize=22)
    plt.plot(VeRed,FeRed,marker='o')

    plt.xlim(0.9*VeRed[0],1.1*VeRed[-1])
    plt.xlabel(r'$V_{e}^{*}$   ($\dfrac{m}{s}$)',fontsize=16)
    plt.ylim(1.2*FeRed[-1],1.2*FeRed[0])
    plt.ylabel(r'$F_{e}^{*}$   (kg)',fontsize=16)
    plt.axhline(0,color='k')

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(1.03*VeRed[2],1.05*FeRed[2],'$x_{cg}$ = '+str(round(Xcg,2))+' m\n$\delta_{t_{e}}$ = '+str(round(deltaTr,3))+'$\degree$',bbox=props,fontsize=16)
    plt.grid()
    return
