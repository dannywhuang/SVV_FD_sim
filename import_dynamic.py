import scipy.io as sio
import pandas as pd


def matToCSV(inputFile,outputFile):
    '''
    Converts .mat to .csv file
    ----
    Input:  inputFile [String]      Name of the .mat file
            outputFile [String]     Name of the .csv file

    Output: None
    '''
    
    raw = sio.loadmat('dynamicData/'+inputFile+'.mat')
    dataFlight = raw['flightdata']
    dataNames = dataFlight.dtype.names

    dict = {}

    for name in dataNames:
        data = dataFlight[name][0,0]['data'][0,0].flatten()
        dict[name] = data
    df = pd.DataFrame(data=dict)
    df.to_csv('dynamicData/'+outputFile+'.csv', index=False)
    return

def sliceTime(fileName,t0,duration):
    '''
    Get data from .csv file starting from t0, until t0+duration
    ----
    Input:  fileName [String]       Name of the .csv file
            t0 [Value]              Starting time
            duration [Value]        Time duration

    Output: dfTime [Dataframe]      Pandas dataframe containing data from t0 until t0+duration
    '''

    df = pd.read_csv('dynamicData/'+fileName+'.csv')
    dfTime = df.loc[(df['time'] >= t0) & (df['time'] <= t0+duration)]

    return dfTime
