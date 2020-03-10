import pandas as pd


def excelToCsv(inputFile):
    '''
    Converting measurement data from excel file to csv file
    ----
    Input:      inputFile [string]                  Excel name file

    Output:     None
    '''

    # Reading the right rows in the excel file
    static1  = pd.read_excel('staticData/'+inputFile+'.xlsx', sheet_name='static1', skiprows=2)
    static1.to_csv('staticData/static1.csv', index=False)

    static2a = pd.read_excel('staticData/'+inputFile+'.xlsx', sheet_name='static2a', skiprows=2)
    static2a.to_csv('staticData/static2a.csv', index=False)

    static2b = pd.read_excel('staticData/'+inputFile+'.xlsx', sheet_name='static2b', skiprows=2)
    static2b.to_csv('staticData/static2b.csv', index=False)

    return 


# excelToCsv('reference')