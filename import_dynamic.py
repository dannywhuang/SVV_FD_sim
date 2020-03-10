import scipy.io as sio
import pandas as pd


def rawToCSV():
    #----------------------------------------------------------------------------------------------
    #
    # Converts matlab.mat to data.csv file
    #
    # Input:    None
    # Output:   None
    #
    #----------------------------------------------------------------------------------------------
    raw = sio.loadmat('dynamicData/matlab.mat')
    dataFlight = raw['flightdata']
    dataNames = dataFlight.dtype.names

    dict = {}

    for name in dataNames:
        data = dataFlight[name][0,0]['data'][0,0].flatten()
        dict[name] = data
    df = pd.DataFrame(data=dict)
    df.to_csv('dynamicData/data.csv', index=False)
    return
