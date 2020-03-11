import pandas as pd
import numpy as np
import os


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


def excelToCSV(inputFile):
    '''
    DESCRIPTION:    Converting measurement data from excel file to csv file
    ========
    INPUT:\n
    ... inputFile [String]:         Excel file name\n

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


def staticMeas(inputFile, dataSet, SI=True):
    '''
    DESCRIPTION:    Get data from .csv file
    ========
    INPUT:\n
    ... inputFile [String]:         Name of the .csv file\n
    ... dataSet [String]:           Name of the excel file (e.g. 'reference' or 'actual' )\n
    ... SI [Condition]:             by default set to SI=True\n

    OUTPUT:\n
    ... df [Dataframe]:             Pandas dataframe containing data
    '''
    if SI == True:
        df = pd.read_csv('staticData/CSV_'+dataSet+'/'+inputFile+'_SI.csv')
    elif SI == False:
        df = pd.read_csv('staticData/CSV_'+dataSet+'/'+inputFile+'.csv')
    else:
        raise ValueError("Incorrect SI input; set it either to '=False', '=True' or leave it empty")
    return df


# excelToCSV('reference')
# convertStaticToSI('reference')





''' Delete this part once understood: to see how the functions work '''
# static1  = staticMeas('static1', 'reference', SI=False)
# static2a = staticMeas('static2a', 'reference', SI=False)
# static2b = staticMeas('static2b', 'reference', SI=False)

# static1_SI  = staticMeas('static1', 'reference')
# static2a_SI = staticMeas('static2a', 'reference')
# static2b_SI = staticMeas('static2b', 'reference')

# print(static1['hp'].to_numpy(), static1_SI['hp'].to_numpy(),'\n')
# print(static2a['Vi'].to_numpy(), static2a_SI['Vi'].to_numpy(),'\n')
# print(static2b['TAT'].to_numpy(), static2b_SI['TAT'].to_numpy(),'\n')

