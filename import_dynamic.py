import scipy.io as sio
import pandas as pd


def matToCSV(inputFile,outputFile):
    #----------------------------------------------------------------------------------------------
    #
    # Converts .mat to .csv file
    #
    # Input:    inputFile [String]      Name of the .mat file
    #           outputFile [String]     Name of the .csv file
    # Output:   None
    #
    #----------------------------------------------------------------------------------------------
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

#matToCSV('reference','reference')