import pandas
import numpy as np

ft  = 0.3048    # [m]
kts = 0.5144    # [m/s]
deg = np.pi/180 # [rad]
lbs = 0.4536    # [kg]

def readStationary(inputFile):
    #----------------------------------------------------------------------------------------------
    #
    # Reading the data of the stationary measurement series, which are saved in a excel file
    #
    # Input:    none
    # Output:   meas1 [array]       Data of CL-CD Series 1: [hp,Vi,alpha,FFl,FFr,Fused,Temp]
    #           measele [array]     Data of Elevator Trim Curve: [hp,Vi,alpha,de,detr,Fe,FFl,FFr,Fused,Temp]
    #           measshift [array]   Data of shift in cg: [Position,Fused]
    #
    #----------------------------------------------------------------------------------------------


    # ----- Stationary measurments CL-CD Series 1 -----

    # Number of measurements
    nr_meas1 = 6
    # Determine the rows that needs to be skipped
    rows_meas1 = list(np.arange(1,24)) + list(np.arange(27+nr_meas1,84))

    # Reading the right rows in the excel file
    read_meas1 = pandas.read_excel('staticData/'+inputFile+'.xlsx',header=1,skiprows=rows_meas1)

    # Transform the columns into lists
    hp    = read_meas1['hp'].tolist()
    Vi    = read_meas1['IAS'].tolist()
    alpha = read_meas1['a'].tolist()
    FFl   = read_meas1['FFl'].tolist()
    FFr   = read_meas1['FFr'].tolist()
    Fused = read_meas1['F. used'].tolist()
    Temp  = read_meas1['TAT'].tolist()

    # Create the output list
    meas1 = [['hp','Vi','alpha','FFl','FFr','Fused','Temp']]

    # Create the rows containing the measurements
    for i in range(2,len(hp)):

        # Transform to SI untis
        hpi    = float(hp[i])*ft        # [m]
        Vii    = float(Vi[i])*kts       # [m/s]
        alphai = float(alpha[i])*deg    # [rad]
        FFli   = float(FFl[i])*lbs/3600 # [kg/s]
        FFri   = float(FFr[i])*lbs/3600 # [kg/s]
        Fusedi = float(Fused[i])*lbs    # [kg]
        Tempi  = float(Temp[i])         # [deg C]

        # Append the row
        meas1.append([hpi,Vii,alphai,FFli,FFri,Fusedi,Tempi])

    # Transform into a numpy array
    meas1 = np.array(meas1)


    # ----- Stationary measurements Elevator Trim Curve -----

    # Number of measurements
    nr_measele = 7
    # Determine the rows that needs to be skipped
    rows_measele = list(np.arange(1,55)) + list(np.arange(58+nr_measele,84))

    # Reading the right rows in the excel file
    read_measele = pandas.read_excel('staticData/'+inputFile+'.xlsx',header=1,skiprows=rows_measele)

    # Transform the columns into lists
    hp    = read_measele['hp'].tolist()
    Vi    = read_measele['IAS'].tolist()
    alpha = read_measele['a'].tolist()
    de    = read_measele['de'].tolist()
    detr  = read_measele['detr'].tolist()
    Fe    = read_measele['Fe'].tolist()
    FFl   = read_measele['FFl'].tolist()
    FFr   = read_measele['FFr'].tolist()
    Fused = read_measele['F. used'].tolist()
    Temp  = read_measele['TAT'].tolist()

    # Create the output list
    measele = [['hp','Vi','alpha','delta','deltaTr','Fe','FFl','FFr','Fused','Temp']]

    # Create the rows containing the measurements
    for i in range(2,len(hp)):

        # Transform to SI untis
        hpi    = float(hp[i])*ft        # [m]
        Vii    = float(Vi[i])*kts       # [m/s]
        alphai = float(alpha[i])*deg    # [rad]
        dei    = float(de[i])*deg       # [rad]
        detri  = float(detr[i])*deg     # [rad]
        Fei    = float(Fe[i])           # [N]
        FFli   = float(FFl[i])*lbs/3600 # [kg/s]
        FFri   = float(FFr[i])*lbs/3600 # [kg/s]
        Fusedi = float(Fused[i])*lbs    # [kg]
        Tempi  = float(Temp[i])         # [deg C]

        # Append the row
        measele.append([hpi,Vii,alphai,dei,detri,Fei,FFli,FFri,Fusedi,Tempi])

    # Transform into a numpy array
    measele = np.array(measele)


    # ----- Shift in center of gravity -----

    # Determine the rows that needs to be skipped
    row_measposition = list(np.arange(1,70)) + list(np.arange(71,84))

    # Reading the right rows in the excel file
    measposition = pandas.read_excel('staticData/'+inputFile+'.xlsx',skiprows=row_measposition)

    # Add the first and second position to a list
    position = [float(measposition['Unnamed: 2']),float(measposition['Unnamed: 7'])]

    # Determine the rows that needs to be skipped
    rows_measshift = list(np.arange(1,72)) + list(np.arange(76,84))

    # Reading the right rows in the excel file
    read_measshift = pandas.read_excel('staticData/'+inputFile+'.xlsx', header=1, skiprows=rows_measshift)

    # Transform the columns into lists
    Fused = read_measshift['F. used'].tolist()

    # Create the output list
    measshift = [['Position','Fused']]

    # Create the rows containing the measurements
    for i in range(1,len(Fused)):

        # Transform to SI untis
        Fusedi = Fused[i]*lbs # [kg]

        # Append the row
        measshift.append([position[i-1],Fusedi])

    # Transform into a numpy array
    measshift = np.array(measshift)

    return meas1,measele,measshift

a = readStationary('reference')