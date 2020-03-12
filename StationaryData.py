import pandas
import numpy as np

ft  = 0.3048    # [m]
kts = 0.5144    # [m/s]
deg = np.pi/180 # [rad]
lbs = 0.4536    # [kg]

def readFirstStationary(file_name='Post_Flight_Datasheet_Flight_1_DD_12_3_2018.xlsx',nr_meas=6,row_start=28):
    #----------------------------------------------------------------------------------------------
    #
    # Reading the data of the first stationary measurement series, which are saved in a excel file
    #
    # Input:    file_name           Name of the excel file
    #           nr_meas = 6         Number of measurements
    #           row_start = 28      Row containing the first measurement
    # Output:   data [array]        Data of the first stationary measurement series: [hp,Vi,alpha,FFl,FFr,Fused,Temp]
    #
    #----------------------------------------------------------------------------------------------

    # Determine the rows that needs to be skipped
    rows = list(np.arange(1, row_start-4)) + list(np.arange(row_start-1+nr_meas, 84))

    # Reading the right rows in the excel file
    read_data = pandas.read_excel(file_name,header=1,skiprows=rows)

    # Transform the columns into lists
    hp    = read_data['hp'].tolist()
    Vi    = read_data['IAS'].tolist()
    alpha = read_data['a'].tolist()
    FFl   = read_data['FFl'].tolist()
    FFr   = read_data['FFr'].tolist()
    Fused = read_data['F. used'].tolist()
    Temp  = read_data['TAT'].tolist()

    # Create the output list
    data = [['hp','Vi','alpha','FFl','FFr','Fused','Temp']]

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
        data.append([hpi,Vii,alphai,FFli,FFri,Fusedi,Tempi])

    # Transform into a numpy array
    data = np.array(data)

    return data


def readSecondStationary(file_name='Post_Flight_Datasheet_Flight_1_DD_12_3_2018.xlsx',nr_meas=7,row_start=59):
    #----------------------------------------------------------------------------------------------
    #
    # Reading the data of the stationary measurement series, which are saved in a excel file
    #
    # Input:    file_name           Name of the excel file
    #           nr_meas = 7         Number of measurements
    #           row_start = 59      Row containing the first measurement
    # Output:   data [array]        Data of the second stationary measurement series: [hp,Vi,alpha,de,detr,Fe,FFl,FFr,Fused,Temp]
    #
    #----------------------------------------------------------------------------------------------

    # Determine the rows that needs to be skipped
    rows = list(np.arange(1, row_start-4)) + list(np.arange(row_start-1+nr_meas,84))

    # Reading the right rows in the excel file
    read_data = pandas.read_excel(file_name,header=1,skiprows=rows)

    # Transform the columns into lists
    hp    = read_data['hp'].tolist()
    Vi    = read_data['IAS'].tolist()
    alpha = read_data['a'].tolist()
    de    = read_data['de'].tolist()
    detr  = read_data['detr'].tolist()
    Fe    = read_data['Fe'].tolist()
    FFl   = read_data['FFl'].tolist()
    FFr   = read_data['FFr'].tolist()
    Fused = read_data['F. used'].tolist()
    Temp  = read_data['TAT'].tolist()

    # Create the output list
    data = [['hp','Vi','alpha','de','detr','Fe','FFl','FFr','Fused','Temp']]

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
        data.append([hpi,Vii,alphai,dei,detri,Fei,FFli,FFri,Fusedi,Tempi])

    # Transform into a numpy array
    data = np.array(data)

    return data


def readcg(file_name='Post_Flight_Datasheet_Flight_1_DD_12_3_2018.xlsx',nr_pax=9,row_start_mass=8,row_block=18,row_shift=71,row_start_meas=75):
    #----------------------------------------------------------------------------------------------
    #
    # Reading the data of the masses and the shift in cg, which are saved in a excel file
    #
    # Input:    file_name                  Name of the excel file
    #           nr_pax = 9                 Number of people aboard
    #           row_start_mass = 8         Row containing the first mass
    #           row_block = 18             Row containing the block fuel mass
    #           row_shift = 71             Row containing the shift
    #           row_start_meas = 75        Row containing the first measurement
    # Output:   data [array]               Data of the second stationary measurement series: [hp,Vi,alpha,de,detr,Fe,FFl,FFr,Fused,Temp]
    #
    #----------------------------------------------------------------------------------------------

    # Determine the rows that needs to be skipped
    rows_mass = list(np.arange(1, row_start_mass-2)) + list(np.arange(row_start_mass-1+nr_pax, 84))
    row_block = list(np.arange(1, row_block-1)) + list(np.arange(row_block, 84))
    row_shift = list(np.arange(1, row_shift-1)) + list(np.arange(row_shift, 84))
    rows_meas = list(np.arange(1,row_start_meas-3)) + list(np.arange(row_start_meas+1,84))

    # Reading the right rows in the excel file
    read_mass = pandas.read_excel(file_name, header=1, skiprows=rows_mass)
    read_block = pandas.read_excel(file_name, skiprows=row_block)
    read_shift = pandas.read_excel(file_name, skiprows=row_shift)
    read_meas = pandas.read_excel(file_name, header=1, skiprows=rows_meas)

    # Transform the columns into lists & get data
    seats = read_mass['Unnamed: 0'].tolist()
    masses = read_mass['mass [kg]'].tolist()
    blockfuel = float(read_block['Unnamed: 3'])*lbs # [kg]
    shift = [float(read_shift['Unnamed: 2']), float(read_shift['Unnamed: 7'])]
    Fused = read_meas['F. used'].tolist()

    # Create the output lists
    mass = [['Seat','Mass']]
    data = [['Position','Fused']]

    # Create the rows containing the measurements
    for i in range(1,len(Fused)):

        # Transform to SI untis
        Fusedi = Fused[i]*lbs # [kg]

        # Append the row
        data.append([shift[i-1],Fusedi])

    for i in range(len(seats)):
        mass.append([seats[i].replace(':',''),masses[i]])

    # Transform into a numpy array
    mass = np.array(mass)
    data = np.array(data)

    return mass,blockfuel,data