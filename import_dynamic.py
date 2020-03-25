import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def matToCSV(inputFile):
    '''
    DESCRIPTION:    Converts .mat to .csv file
    ========
    INPUT:\n
    ... inputFile [String]:         Name of the .mat file
    ... outputFile [String]:        Name of the .csv file
    OUTPUT:\n
    ... None
    '''

    raw = sio.loadmat('dynamicData/'+inputFile+'.mat')
    dataFlight = raw['flightdata']
    dataNames = dataFlight.dtype.names

    dict = {}

    for name in dataNames:
        data = dataFlight[name][0,0]['data'][0,0].flatten()
        dict[name] = data
    df = pd.DataFrame(data=dict)
    df.to_csv('dynamicData/'+inputFile+'.csv', index=False)
    return


def convertDynToSI(inputFile):
    '''
    DESCRIPTION:    Converts dynamic flight data units to SI units
    ========
    INPUT:\n
    ... inputFile [String]:         Name of the .csv file to be converted to SI units\n
    OUTPUT:\n
    ... None
    '''
    
    df = pd.read_csv('dynamicData/'+inputFile+'.csv')
    degList = ['vane_AOA','elevator_dte','delta_a','delta_e','delta_r','Ahrs1_Roll','Ahrs1_Pitch','Ahrs1_bRollRate','Ahrs1_bPitchRate','Ahrs1_bYawRate','Gps_lat','Gps_long']
    for degName in degList:
        df[degName] = np.radians(df[degName])           #convert degrees or degrees/s to radians or radians/s

    lbsHrList = ['lh_engine_FMF','rh_engine_FMF']
    for lbsHrName in lbsHrList:
        df[lbsHrName] = df[lbsHrName]*0.45359237/3600   # convert lbs/hr to kg/s

    degCelciusList = ['lh_engine_itt','rh_engine_itt','Dadc1_sat','Dadc1_tat']
    for degCelciusName in degCelciusList:
        df[degCelciusName] = df[degCelciusName]+273.15  # convert degrees Celsius to Kelvin

    psiList = ['lh_engine_OP','rh_engine_OP']
    for psiName in psiList:
        df[psiName] = df[psiName]*6894.75729            # convert psi to Pascals

    lbsList = ['lh_engine_FU','rh_engine_FU']
    for lbsName in lbsList:
        df[lbsName] = df[lbsName]*0.45359237            # convert lbs to kg

    gList = ['Ahrs1_bLongAcc','Ahrs1_bLatAcc','Ahrs1_bNormAcc','Ahrs1_aHdgAcc','Ahrs1_xHdgAcc','Ahrs1_VertAcc']
    for gName in gList:
        df[gName] = df[gName]*9.81                      # convert g to m/s^2

    ftList = ['Dadc1_alt','Dadc1_bcAlt']
    for ftName in ftList:
        df[ftName] = df[ftName]*0.3048                  # convert feet to meter

    knotsList = ['Dadc1_cas','Dadc1_tas']
    for knotsName in knotsList:
        df[knotsName] = df[knotsName]*0.514444444       # convert knots to m/s

    ftMinList = ['Dadc1_altRate']
    for ftMinName in ftMinList:
        df[ftMinName] = df[ftMinName]*0.3048/60         # convert feet/min to meter/s

    df.to_csv('dynamicData/'+inputFile+'_SI.csv',index=False)
    return


def sliceTime(fileName,t0,duration,SI=True):
    '''
    DESCRIPTION:    Get data from .csv file starting from t0, until t0+duration
    ========
    INPUT:\n
    ... fileName [String]:          Name of the .csv file\n
    ... t0 [Value]:                 Starting time\n
    ... duration [Value]:           Time duration\n
    ... SI [Boolean]:               True for SI, false for non SI\n
    OUTPUT:\n
    ... dfTime [Dataframe]:         Pandas dataframe containing data from t0 until t0+duration
    '''
    if SI==True:
        df = pd.read_csv('dynamicData/'+fileName+'_SI.csv')
        dfTime = df.loc[(df['time'] >= t0) & (df['time'] <= t0+duration)]
    elif SI==False:
        df = pd.read_csv('dynamicData/' + fileName + '.csv')
        dfTime = df.loc[(df['time'] >= t0) & (df['time'] <= t0 + duration)]
    else:
        raise ValueError("Enter SI = True or SI = False")
    return dfTime


def dynamicMeas(fileName,SI=True):
    '''
    DESCRIPTION:    Get data from .csv file starting from t0, until t0+duration
    ========
    INPUT:\n
    ... fileName [String]:          Name of the .csv file\n
    ... SI [Boolean]:               True for SI, false for non SI\n
    OUTPUT:\n
    ... df [Dataframe]:             Pandas dataframe containing data
    '''
    if SI==True:
        df = pd.read_csv('dynamicData/'+fileName+'_SI.csv')
    elif SI==False:
        df = pd.read_csv('dynamicData/' + fileName + '.csv')
    else:
        raise ValueError("Enter SI = True or SI = False")
    return df


def plotData(inputFile,t0,duration):
    # import data
    actualDf = sliceTime(inputFile,t0,duration,True)
    time = actualDf['time'].to_numpy()
    speed = actualDf['Dadc1_tas'].to_numpy()
    delta_e = actualDf['delta_e'].to_numpy()
    delta_a = actualDf['delta_a'].to_numpy()
    delta_r = actualDf['delta_r'].to_numpy()
    alpha = actualDf['vane_AOA'].to_numpy()
    theta = actualDf['Ahrs1_Pitch'].to_numpy()
    pitchRate = actualDf['Ahrs1_bPitchRate'].to_numpy()
    roll =actualDf['Ahrs1_Roll'].to_numpy()
    rollRate =actualDf['Ahrs1_bRollRate'].to_numpy()
    yawRate = actualDf['Ahrs1_bYawRate'].to_numpy()

    # plot speed, and inputs
    fig, axs = plt.subplots(4, 1, figsize=(16, 9), dpi=100)
    plt.suptitle('Dynamic data versus time: %s')
    axs[0].plot(time, speed, label="Speed")
    axs[0].set_ylabel(r'$V_{TAS}$ [m/s]', fontsize=12.0)
    axs[0].grid()

    axs[1].plot(time, delta_e, label="elevator deflection")
    axs[1].set_ylabel(r'$\delta_{e}$ [rad]', fontsize=12.0)
    axs[1].grid()

    axs[2].plot(time, delta_a, label="aileron deflection")
    axs[2].set_ylabel(r'$\delta_{a}$ [rad]', fontsize=12.0)
    axs[2].grid()

    axs[3].plot(time, delta_r, label="rudder deflection")
    axs[3].set_ylabel(r'$\delta_{r}$ [rad]', fontsize=12.0)
    axs[3].set_xlabel('time [s]', fontsize=12.0)
    axs[3].grid()
    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[0].tick_params(axis='both', which='minor', labelsize=12)
    axs[1].tick_params(axis='both', which='major', labelsize=12)
    axs[1].tick_params(axis='both', which='minor', labelsize=12)
    axs[2].tick_params(axis='both', which='major', labelsize=12)
    axs[2].tick_params(axis='both', which='minor', labelsize=12)
    axs[3].tick_params(axis='both', which='major', labelsize=12)
    axs[3].tick_params(axis='both', which='minor', labelsize=12)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


    # plot symmetric data
    fig, axs = plt.subplots(3, 1, figsize=(16, 9), dpi=100)
    plt.suptitle('Symmetric data versus time: %s')

    axs[0].plot(time, alpha, label="alpha")
    axs[0].set_ylabel(r'$\alpha$ [rad]', fontsize=12.0)
    axs[0].grid()

    axs[1].plot(time, theta, label="theta")
    axs[1].set_ylabel(r'$\theta$ [rad]', fontsize=12.0)
    axs[1].grid()

    axs[2].plot(time, pitchRate, label="pitch rate")
    axs[2].set_ylabel(r'pitch rate [rad]', fontsize=12.0)
    axs[2].grid()

    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[0].tick_params(axis='both', which='minor', labelsize=12)
    axs[1].tick_params(axis='both', which='major', labelsize=12)
    axs[1].tick_params(axis='both', which='minor', labelsize=12)
    axs[2].tick_params(axis='both', which='major', labelsize=12)
    axs[2].tick_params(axis='both', which='minor', labelsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # plot asymmetric data
    fig, axs = plt.subplots(3, 1, figsize=(16, 9), dpi=100)
    plt.suptitle('Asymmetric data versus time: %s')

    axs[0].plot(time, roll, label="roll")
    axs[0].set_ylabel(r'roll [rad]', fontsize=12.0)
    axs[0].grid()

    axs[1].plot(time, rollRate, label="roll rate")
    axs[1].set_ylabel(r'roll rate [rad/s]', fontsize=12.0)
    axs[1].grid()

    axs[2].plot(time, yawRate, label="yaw rate")
    axs[2].set_ylabel(r'yaw rate [rad/s]', fontsize=12.0)
    axs[2].grid()

    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[0].tick_params(axis='both', which='minor', labelsize=12)
    axs[1].tick_params(axis='both', which='major', labelsize=12)
    axs[1].tick_params(axis='both', which='minor', labelsize=12)
    axs[2].tick_params(axis='both', which='major', labelsize=12)
    axs[2].tick_params(axis='both', which='minor', labelsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()

    return