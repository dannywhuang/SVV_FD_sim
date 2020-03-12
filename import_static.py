import pandas as pd
import numpy as np
import os
from numpy import sqrt

from main import ParametersOld


# To create files
def createFolder(directory):
    '''
    DESCRIPTION:    Create an empty folder
    ========
    INPUT:\n
    ... directory [string]:         Location and name of created folder\n

    OUTPUT:\n
    ... None
    '''

    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
    return


# To create files
def excelToCSV(inputFile):
    '''
    DESCRIPTION:    Converting measurement data from excel file to csv file
    ========
    INPUT:\n
    ... inputFile [String]:         Excel file name; choose between 'reference' or 'actual'\n

    OUTPUT:\n
    ... None
    '''

    # Creating empty folder to store created CSV files in
    createFolder('staticData/CSV_'+inputFile)

    # Reading the right rows in the excel file
    static1  = pd.read_excel('staticData/'+inputFile+'.xlsx', sheet_name='static1', skiprows=2)
    static1.to_csv('staticData/CSV_'+inputFile+'/static1.csv', index=False)

    static2a = pd.read_excel('staticData/'+inputFile+'.xlsx', sheet_name='static2a', skiprows=2)
    static2a.to_csv('staticData/CSV_'+inputFile+'/static2a.csv', index=False)

    static2b = pd.read_excel('staticData/'+inputFile+'.xlsx', sheet_name='static2b', skiprows=2)
    static2b.to_csv('staticData/CSV_'+inputFile+'/static2b.csv', index=False)
    return 


# To create files
def convertStaticToSI(inputFile):
    '''
    DESCRIPTION:    Converts static measurement series units to SI units
    ========
    INPUT:\n
    ... inputFile [String]:         Excel file name containing data to convert to SI units\n

    OUTPUT:\n
    ... None
    '''

    df1  = pd.read_csv('staticData/CSV_'+inputFile+'/static1.csv')
    df2a = pd.read_csv('staticData/CSV_'+inputFile+'/static2a.csv')
    df2b = pd.read_csv('staticData/CSV_'+inputFile+'/static2b.csv')

    ftList = ['hp']
    for ftName in ftList:
        df1[ftName] = df1[ftName] * 0.3048                  # convert feet to meter
        df2a[ftName] = df2a[ftName] * 0.3048
        df2b[ftName] = df2b[ftName] * 0.3048

    knotsList = ['Vi']
    for knotsName in knotsList:
        df1[knotsName] = df1[knotsName] * 0.514444444       # convert knots to m/s
        df2a[knotsName] = df2a[knotsName] * 0.514444444
        df2b[knotsName] = df2b[knotsName] * 0.514444444

    degList = ['delta', 'deltaTr']
    for degName in degList:             
        df2a[degName] = np.radians(df2a[degName])           # convert degrees or degrees/s to radians or radians/s
        df2b[degName] = np.radians(df2b[degName])

    lbsHrList = ['FFl', 'FFr']
    for lbsHrName in lbsHrList:
        df1[lbsHrName] = df1[lbsHrName] * 0.45359237/3600   # convert lbs/hr to kg/s
        df2a[lbsHrName] = df2a[lbsHrName] * 0.45359237/3600
        df2b[lbsHrName] = df2b[lbsHrName] * 0.45359237/3600

    lbsList = ['F. Used']
    for lbsName in lbsList:
        df1[lbsName] = df1[lbsName] * 0.45359237            # convert lbs to kg
        df2a[lbsName] = df2a[lbsName] * 0.45359237
        df2b[lbsName] = df2b[lbsName] * 0.45359237

    degCelciusList = ['TAT']
    for degCelciusName in degCelciusList:
        df1[degCelciusName] = df1[degCelciusName] + 273.15  # convert degrees Celsius to Kelvin
        df2a[degCelciusName] = df2a[degCelciusName] + 273.15
        df2b[degCelciusName] = df2b[degCelciusName] + 273.15

    df1.to_csv('staticData/CSV_'+inputFile+'/static1_SI.csv',index=False)
    df2a.to_csv('staticData/CSV_'+inputFile+'/static2a_SI.csv',index=False)
    df2b.to_csv('staticData/CSV_'+inputFile+'/static2b_SI.csv',index=False)
    return


# To create files
def thrustToDAT(inputFile, SI=True, standard=False):
    '''
    DESCRIPTION:    Creates a dat file names 'matlab.dat' containing all necessary data to be used for the thrust calculations. This data is obtained from the created csv files and some parameters will be calculated before putting everyting in the created .dat file.
    ========
    INPUT:\n
    ... inputFile [String]:         Excel file name; choose between 'reference' or 'actual'\n
    ... SI [Condition]:             By default set to SI=True\n                         

    OUTPUT:\n
    ... None
    '''

    param = ParametersOld()

    thrustData1 = {}
    thrustData2a = {}
    thrustData2b = {}

    # pressure altitude, hp
    dataNames = ['hp','Mach','DeltaTisa','FFl','FFr']

    for name in dataNames:
        if name in ['hp','FFl','FFr']:
            if standard == True and name in ['FFl','FFr']:
                len1   = np.size(staticMeas('reference', 'static1', SI)[name].to_numpy())
                data1  = np.ones(len1)*param.ms
                len2a  = np.size(staticMeas('reference', 'static2a', SI)[name].to_numpy())
                data2a = np.ones(len2a)*param.ms
                len2b  = np.size(staticMeas('reference', 'static2b', SI)[name].to_numpy())
                data2b = np.ones(len2b)*param.ms
            else:
                data1  = staticMeas('reference', 'static1', SI)[name].to_numpy()
                data2a = staticMeas('reference', 'static2a', SI)[name].to_numpy()
                data2b = staticMeas('reference', 'static2b', SI)[name].to_numpy()
        
        elif name in ['Mach','DeltaTisa']:
            data1  = staticFlightCondition('reference', 'static1', SI)[name].to_numpy()
            data2a = staticFlightCondition('reference', 'static2a', SI)[name].to_numpy()
            data2b = staticFlightCondition('reference', 'static2b', SI)[name].to_numpy()
        thrustData1[name]  = data1
        thrustData2a[name] = data2a
        thrustData2b[name] = data2b
    df1  = pd.DataFrame(data=thrustData1)
    df2a = pd.DataFrame(data=thrustData2a)
    df2b = pd.DataFrame(data=thrustData2b)

    if standard == True:
        createFolder('staticData/thrust_'+inputFile+'/static1/standard')
        createFolder('staticData/thrust_'+inputFile+'/static2a/standard')
        createFolder('staticData/thrust_'+inputFile+'/static2b/standard')

        df1.to_csv('staticData/thrust_'+inputFile+'/static1/standard/matlab.dat',index=False,sep=' ',header=False)
        df2a.to_csv('staticData/thrust_'+inputFile+'/static2a/standard/matlab.dat',index=False,sep=' ',header=False)
        df2b.to_csv('staticData/thrust_'+inputFile+'/static2b/standard/matlab.dat',index=False,sep=' ',header=False)
    elif standard == False:
        createFolder('staticData/thrust_'+inputFile+'/static1')
        createFolder('staticData/thrust_'+inputFile+'/static2a')
        createFolder('staticData/thrust_'+inputFile+'/static2b')

        df1.to_csv('staticData/thrust_'+inputFile+'/static1/matlab.dat',index=False,sep=' ',header=False)
        df2a.to_csv('staticData/thrust_'+inputFile+'/static2a/matlab.dat',index=False,sep=' ',header=False)
        df2b.to_csv('staticData/thrust_'+inputFile+'/static2b/matlab.dat',index=False,sep=' ',header=False)
    return


# Return data from files
def staticMeas(inputFile, dataSet, SI=True):
    '''
    DESCRIPTION:    Get data from .csv file
    ========
    INPUT:\n
    ... inputFile [String]:         Name of the excel file (e.g. 'reference' or 'actual' )\n
    ... dataSet [String]:           Name of data file (e.g. 'static1', 'static2a' or 'static2b'\n
    ... SI [Condition]:             By default set to SI=True\n

    OUTPUT:\n
    ... df [Dataframe]:             Pandas dataframe containing data
    '''

    if SI == True:
        df = pd.read_csv('staticData/CSV_'+inputFile+'/'+dataSet+'_SI.csv')
    elif SI == False:
        df = pd.read_csv('staticData/CSV_'+inputFile+'/'+dataSet+'.csv')
    else:
        raise ValueError("Incorrect SI input; set it either to '=False', '=True' or leave it empty")
    return df


# Return data from files
def staticFlightCondition(inputFile, dataSet, SI=True):
    '''
    DESCRIPTION:    This function calculates flight conditions using the measurement data. 
    ========
    INPUT:\n
    ... inputFile [String]:             Name of excel file; choose between 'reference' or 'actual'\n
    ... dataSet [String]:               Name of data set; choose between 'static1', 'static2a' or 'static2b'\n
    ... SI [Condition]:                 By default set to SI=True\n

    OUTPUT:\n
    ... staticFlightCond [Dataframe]:   Pandas dataframe containing flight condition values that are not measured.
    '''

    meas = staticMeas(inputFile, dataSet, SI)
    param = ParametersOld()

    # Constant values
    pres0 = param.pres0
    rho0  = param.rho0 
    Temp0 = param.Temp0 
    g0    = param.g 
    Ws    = param.Ws

    gamma = param.gamma
    lamb  = param.lamb 
    R     = param.R

    Vc       = meas['Vi'].to_numpy()

    TempMeas = meas['TAT'].to_numpy()
    hp       = meas['hp'].to_numpy()

    # Function values
    W = 123#weight function ...

    # Calculation
    pres = pres0 * (1 + lamb * hp / Temp0) ** (- g0 / (lamb * R))
    Mach = sqrt(2/(gamma - 1) * ((1 + pres0/pres * ((1 + (gamma - 1)/(2 * gamma) * rho0/pres0 * Vc**2)**( gamma/(gamma - 1) ) - 1))**( (gamma - 1)/gamma ) - 1))
    Temp = TempMeas / ( 1 + (lamb - 1)/2 * Mach**2 )
    a = sqrt(gamma * R * Temp)
    Vt = Mach * a
    rho = pres / (R * Temp)
    Ve = Vt * sqrt(rho / rho0)

    VeRed = Ve * sqrt( Ws / W )

    Tisa = Temp0 + lamb * meas['hp'].to_numpy()
    DeltaTisa = Temp - Tisa

    staticFlightCond = {}
    dataNames = ['pres','Mach','Temp','a','Vt','rho','Ve','VeRed','DeltaTisa']
    for name in dataNames:
        staticFlightCond[name] = locals()[name]
    staticFlightCond = pd.DataFrame(data=staticFlightCond)
    return staticFlightCond


# Return data from files
def staticThrust(inputFile, dataSet, standard=False):
    '''
    DESCRIPTION:    This function reads the created thrust file. 
    ========
    INPUT:\n
    ... inputFile [String]:             Name of excel file; choose between 'reference' or 'actual'\n
    ... dataSet [String]:               Name of data set; choose between 'static1', 'static2a' or 'static2b'\n

    OUTPUT:\n
    ... df [Dataframe]:                 Pandas dataframe containing Thrust of the left and right engine & the total Thrust.
    '''

    if standard == True:
        df = pd.read_csv('staticData/thrust_'+inputFile+'/'+dataSet+'/standard/thrust.dat',sep='\t')
        df.columns = ['Tpl','Tpr']
        df['Tp']   = df['Tpl'].to_numpy() + df['Tpr'].to_numpy()
        # df['Tcs']  = df['Tp']
    elif standard == False:
        df = pd.read_csv('staticData/thrust_'+inputFile+'/'+dataSet+'/thrust.dat',sep='\t')
        df.columns = ['Tpl','Tpr']
        df['Tp']   = df['Tpl'].to_numpy() + df['Tpr'].to_numpy()
        # df['Tc']   = df['Tp']
    return df


''' Create data files from the provided/measured data '''
# excelToCSV('reference')
# convertStaticToSI('reference')
# thrustToDAT('reference', SI=True, standard=False)
# thrustToDAT('reference', SI=True, standard=True)







''' Delete this part once understood: to see how the functions work '''
# static1  = staticMeas('static1', 'reference', SI=False)
# static2a = staticMeas('static2a', 'reference', SI=False)
# static2b = staticMeas('static2b', 'reference', SI=False)

# static1_SI  = staticMeas('static1', 'reference')
# static2a_SI = staticMeas('static2a', 'reference')
# static2b_SI = staticMeas('static2b', 'reference')

# staticCond1  = staticFlightCondition('static1', 'reference', SI=True)
# staticCond2a = staticFlightCondition('static2a', 'reference', SI=True)
# staticCond2b = staticFlightCondition('static2b', 'reference', SI=True)

# staticThrust1  = staticThrust('reference','static1',standard=False)
# staticThrust2a = staticThrust('reference','static2a',standard=False)
# staticThrust2b = staticThrust('reference','static2b',standard=False)


# print('pressure altitude, static1: ',static1['hp'].to_numpy(),'/ pressure altitude, static1 in SI: ',static1_SI['hp'].to_numpy(),'\n')
# print('indicated airspeed, static2a: ',static2a['Vi'].to_numpy(),'/ indicated airspeed, static2a in SI: ', static2a_SI['Vi'].to_numpy(),'\n')
# print('meassured air temp, static2b: ',static2b['TAT'].to_numpy(),'/ meassured air temp, static2b in SI: ', static2b_SI['TAT'].to_numpy(),'\n')

# print('true airspeed, static1 in SI: ',staticCond1['Vt'].to_numpy(),'/ Mach number, static1 in SI: ', staticCond1['Mach'].to_numpy(),'\n')
# print('reduced equivalent airspeed, static2a in SI: ',staticCond2a['VeRed'].to_numpy(),'/ local pressure, static2a in SI: ', staticCond2a['pres'].to_numpy(),'\n')
# print('local air density, static2b in SI: ',staticCond2b['rho'].to_numpy(),'/ corrected local temperature, static2b in SI: ', staticCond2b['Temp'].to_numpy(),'\n')

# print('local thrust left engine, static 1 in [N?]: ',staticThrust1['Tpl'].to_numpy(),'/ local total thrust, static 1 in [N?]: ', staticThrust1['Tp'].to_numpy(),'\n')
# print('local thrust right engine, static 2a in [N?]: ',staticThrust2a['Tpr'].to_numpy(),'/ local total thrust, static 2a in [N?]: ', staticThrust2a['Tp'].to_numpy(),'\n')
# print('local thrust left engine, static 2b in [N?]: ',staticThrust2b['Tpl'].to_numpy(),'/ local total thrust, static 2b in [N?]: ', staticThrust2b['Tp'].to_numpy(),'\n')