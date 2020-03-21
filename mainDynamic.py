#Initial numerical model before making improvements to parameters
import numpy as np
from math import cos, pi, sin, tan, pow
import control.matlab as ml
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

import import_dynamic as imDyn
import import_weight as imWeight

# global constants
g = 9.81


def sampleFunction(param):
    '''
    DESCRIPTION:    Function description
    ========
    INPUT:\n
    ... param [Type]:               Parameter description\n
    ... param [Type]:               Parameter description\n

    OUTPUT:\n
    ... param [Type]:               Parameter description
    '''

    s = param.b+param.S+param.c

    return s


def calcEigenShortPeriod(param):        #Verified by Danny
    '''
    DESCRIPTION:    Calculate eigenvalues of state space matrix for short period
    ========
    INPUT:\n
    ... param [Class]:               Class containing aerodynamic and stability parameters, same parameters as in Cit_par.py\n

    OUTPUT:\n
    ... lambdac [Array]:            Array with eigen values for analytical model short period motion
    '''
    # Short period
    A = 2*param.muc*param.KY2*(2*param.muc-param.CZadot)
    B = -2*param.muc*param.KY2*param.CZa-(2*param.muc+param.CZq)*param.Cmadot-(2*param.muc-param.CZadot)*param.Cmq
    C = param.CZa*param.Cmq-(2*param.muc+param.CZq)*param.Cma
    p = [A,B,C]
    lambdac = np.roots(p)

    return lambdac


def calcEigenPhugoid(param): # For some reason the eigenvalues are a lot different from state space eigenvalues
    '''
    DESCRIPTION:    Calculate eigenvalues of state space matrix for phugoid
    ========
    INPUT:\n
    ... param [Class]:               Class containing aerodynamic and stability parameters, same parameters as in Cit_par.py\n

    OUTPUT:\n
    ... lambdac [Array]:            Array with eigen values for analytical model phugoid motion\n
    ... lambdacSimp [Array]:        Array with eigen values for simplified anayltical model phugoid motion
    '''

    # Phugoid
    A = 2*param.muc*(param.CZa*param.Cmq-2*param.muc*param.Cma)
    B = 2*param.muc*(param.CXu*param.Cma - param.Cmu*param.CXa) + param.Cmq*(param.CZu *param.CXa - param.CXu*param.CZa)
    C = param.CZ0*(param.Cmu*param.CZa - param.CZu*param.Cma)
    p = [A,B,C]
    lambdac = np.roots(p)

    # Simplified phugoid
    A2 = -4*param.muc**2
    B2 = 2*param.muc*param.CXu
    C2 = -param.CZu*param.CZ0
    p2 = [A2,B2,C2]
    lambdacSimp = np.roots(p2)

    return lambdac, lambdacSimp


def calcEigenDutchRoll(param): # Verified by Danny
    '''
    DESCRIPTION:    Calculate eigenvalues of state space matrix for dutch roll
    ========
    INPUT:\n
    ... param [Class]:               Class containing aerodynamic and stability parameters, same parameters as in Cit_par.py\n

    OUTPUT:\n
    ... lambdac [Array]:            Array with eigen values for analytical model dutch roll motion\n
    ... lambdacSimp [Array]:        Array with eigen values for simplified anayltical model dutch roll motion
    '''
    A = 8*param.mub**2*param.KZ2
    B = -2*param.mub*(param.Cnr + 2*param.KZ2*param.CYb)
    C = 4*param.mub*param.Cnb + param.CYb*param.Cnr
    lambdac = np.roots([A, B, C])

    # Simplified Dutch Roll
    A2 = -2*param.mub*param.KZ2
    B2 = 0.5*param.Cnr
    C2 = -param.Cnb
    lambdacSimp = np.roots([A2, B2, C2])

    return lambdac, lambdacSimp


def calcEigenAperRoll(param):   # Verified by Danny
    '''
    DESCRIPTION:    Calculate eigenvalues of state space matrix for short period
    ========
    INPUT:\n
    ... param [Class]:              Class containing aerodynamic and stability parameters, same parameters as in Cit_par.py\n

    OUTPUT:\n
    ... lambdac [Value]:            Eigenvalue for analytical model short period motion
    '''

    # Aperiodic Roll
    lambdac = param.Clp/(4*param.mub*param.KX2)

    return lambdac


def calcEigenSpiral(param): # Verified by Danny
    '''
    DESCRIPTION:    Calculate eigenvalues of state space matrix for spiral
    ========
    INPUT:\n
    ... param [Class]:               Class containing aerodynamic and stability parameters, same parameters as in Cit_par.py\n

    OUTPUT:\n
    ... lambdac [Array]:            Array with eigen values for analytical model spiral motion
    '''

    # Spiral
    lambdac = ((2*param.CL*(param.Clb*param.Cnr-param.Cnb*param.Clr))/(param.Clp*(param.CYb*param.Cnr+4*param.mub*param.Cnb)-param.Cnp*(param.CYb*param.Clr+4*param.mub*param.Clb)))

    return lambdac


def calcResponse(t0,duration,fileName,StateSpace,param,SI=True):
    '''
    DESCRIPTION:    Calculate responses using dynamic measurement data
    ========
    INPUT:\n
    ... t0 [Value]:                 The time at which the response starts\n
    ... duration [Value]:           The duration of the response\n
    ... fileName [String]:          Name of dynamic measurement data file, so reference or flighttest\n
    ... param [Class]:              Class with paramaters of aircraft\n

    OUTPUT:\n
    ... XoutS [Array]:              Array with state variable responses for symmetric\n
    ... YoutS [Array]:              Array with output variable responses for symmetric\n
    ... XoutA [Array]:              Array with state variable responses for asymmetric\n
    ... YoutS [Array]:              Array with output variable responses for asymmetric
    '''

    # get state matrices using aircraft parameters
    As = StateSpace.As
    Bs = StateSpace.Bs
    Cs = StateSpace.Cs
    Ds = StateSpace.Ds
    Aa = StateSpace.Aa
    Ba = StateSpace.Ba
    Ca = StateSpace.Ca
    Da = StateSpace.Da

    # create state space models
    stateSpaceS = ml.ss(As,Bs,Cs,Ds)
    stateSpaceA = ml.ss(Aa,Ba,Ca,Da)

    # get data from the dynamic measurement file
    dfTime = imDyn.sliceTime(fileName,t0,duration,SI)
    time = dfTime['time']
    delta_a = dfTime['delta_a'].to_numpy()
    delta_e = dfTime['delta_e'].to_numpy()
    delta_r = dfTime['delta_r'].to_numpy()

    # input variables
    us = delta_e
    ua = np.vstack((-delta_a,-delta_r)).T

    a0 = dfTime['vane_AOA'].to_numpy()[0]
    a_stab0 = dfTime['vane_AOA'].to_numpy()[0] - a0

    theta0 = a0 - dfTime['Ahrs1_Pitch'].to_numpy()[0]
    theta_stab0 = dfTime['Ahrs1_Pitch'].to_numpy()[0]-theta0
    q0 = dfTime['Ahrs1_bPitchRate'].to_numpy()[0]*(param.c/param.V0)


    # initial condition
    x0s = [0 , a_stab0 , theta_stab0 , q0]
    # x0a = [0 , dfTime['Ahrs1_Roll'].to_numpy()[0] ,(dfTime['Ahrs1_bRollRate'].to_numpy()[0]*param.b)/(2*param.V0)  , (dfTime['Ahrs1_bYawRate'].to_numpy()[0]*param.b)/(2*param.V0)]
    x0a = [0 , 0 , (dfTime['Ahrs1_bRollRate'].to_numpy()[0]*param.b)/(2*param.V0)  , (dfTime['Ahrs1_bYawRate'].to_numpy()[0]*param.b)/(2*param.V0)]
    # calculate responses
    YoutS, tS, XoutS = ml.lsim(stateSpaceS, us, time, x0s)
    YoutA, tA, XoutA = ml.lsim(stateSpaceA, ua, time, x0a)
    
    return tS, XoutS, YoutS, tA, XoutA, YoutA


def calcStepResponse(t0, duration, fileName, StateSpace, param, SI=True):
    '''
    DESCRIPTION:    Calculate responses using dynamic measurement data
    ========
    INPUT:\n
    ... t0 [Value]:                 The time at which the response starts\n
    ... duration [Value]:           The duration of the response\n
    ... fileName [String]:          Name of dynamic measurement data file, so reference or flighttest\n
    ... param [Class]:              Class with paramaters of aircraft\n

    OUTPUT:\n
    ... XoutS [Array]:              Array with state variable responses for symmetric\n
    ... YoutS [Array]:              Array with output variable responses for symmetric\n
    ... XoutA [Array]:              Array with state variable responses for asymmetric\n
    ... YoutS [Array]:              Array with output variable responses for asymmetric
    '''

    # get state matrices using aircraft parameters
    As = StateSpace.As
    Bs = StateSpace.Bs
    Cs = StateSpace.Cs
    Ds = StateSpace.Ds
    Aa = StateSpace.Aa
    Ba = StateSpace.Ba
    Ca = StateSpace.Ca
    Da = StateSpace.Da

    # create state space models
    stateSpaceS = ml.ss(As, Bs, Cs, Ds)
    stateSpaceA = ml.ss(Aa, Ba, Ca, Da)

    # get data from the dynamic measurement file
    dfTime = imDyn.sliceTime(fileName, t0, duration, SI)
    time = dfTime['time']

    # input variables
    a0 = dfTime['vane_AOA'].to_numpy()[0]
    a_stab0 = dfTime['vane_AOA'].to_numpy()[0] - a0

    theta_stab0 = a0 - dfTime['Ahrs1_Pitch'].to_numpy()[0]
    q0 = dfTime['Ahrs1_bPitchRate'].to_numpy()[0] * (param.c / param.V0)

    # initial condition
    x0s = [0, 0,0,0]
    # x0a = [0 , dfTime['Ahrs1_Roll'].to_numpy()[0] ,(dfTime['Ahrs1_bRollRate'].to_numpy()[0]*param.b)/(2*param.V0)  , (dfTime['Ahrs1_bYawRate'].to_numpy()[0]*param.b)/(2*param.V0)]
    x0a = [0, 0, 0,0]
    # calculate responses
    YoutS, tS = ml.step(stateSpaceS, time, x0s)
    YoutA, tA = ml.step(stateSpaceA, time, x0a)

    return tS, YoutS, tA, YoutA


def stateSpace(param):
    '''
    DESCRIPTION:    Calculate state-space matrices for symmetric and asymmetric equations of motion
    ========
    INPUT:\n
    ... param [Class]:                          Class containing aerodynamic and stability parameters, same parameters as in Cit_par.py\n
    
    OUTPUT:\n
    ... ss [Class]:                             Class containing state-space matrices A,B,C,D and eigenvalues for (s)ymmetric and (a)symmetric equations of moton
    '''

    C1s = np.matrix([[ -2*param.muc*(param.c/param.V0) , 0 , 0 , 0 ],
                     [ 0 , (param.CZadot-2*param.muc)*(param.c/param.V0) , 0 , 0 ],
                     [ 0 , 0 , -(param.c/param.V0) , 0 ],
                     [ 0 , param.Cmadot*(param.c/param.V0) , 0 , -2*param.muc*param.KY2*(param.c/param.V0)]])

    C2s = np.matrix([[ param.CXu , param.CXa , param.CZ0 , param.CXq ],
                     [ param.CZu , param.CZa, -param.CX0 , (param.CZq+2*param.muc)],
                     [ 0 , 0 , 0 , 1 ],
                     [ param.Cmu , param.Cma , 0 , param.Cmq]])

    C3s = np.matrix([[ param.CXde ],
                     [ param.CZde ],
                     [ 0 ],
                     [ param.Cmde]])

    C1sInverse = np.linalg.inv(C1s)
    As = -np.matmul(C1sInverse,C2s)
    Bs = -np.matmul(C1sInverse,C3s)
    Cs = np.eye(4)
    Ds = np.zeros((4,1))

    C1a = np.matrix([[ (param.CYbdot-2*param.mub)*(param.b/param.V0) , 0 , 0 , 0 ],
                     [ 0 , -(1/2)*(param.b/param.V0) , 0 , 0 ],
                     [ 0 , 0 , -4*param.mub*param.KX2*(param.b/param.V0) , 4*param.mub*param.KXZ*(param.b/param.V0) ],
                     [ param.Cnbdot*(param.b/param.V0) , 0 , 4*param.mub*param.KXZ*(param.b/param.V0) , -4*param.mub*param.KZ2*(param.b/param.V0) ]])

    C2a = np.matrix([[ param.CYb, param.CL , param.CYp , (param.CYr-4*param.mub) ],
                     [ 0    , 0 , 1 , 0 ],
                     [ param.Clb , 0 , param.Clp , param.Clr ],
                     [ param.Cnb , 0 , param.Cnp , param.Cnr ]])

    C3a = np.matrix([[ param.CYda , param.CYdr ],
                     [ 0 , 0 ],
                     [ param.Clda , param.Cldr ],
                     [ param.Cnda , param.Cndr ]])

    C1aInverse = np.linalg.inv(C1a)
    Aa = -np.matmul(C1aInverse, C2a)
    Ba = -np.matmul(C1aInverse, C3a)
    Ca = np.eye(4)
    Da = np.zeros((4, 2))

    Eigs = np.linalg.eig(As)[0]*(param.c/param.V0)
    Eiga = np.linalg.eig(Aa)[0]*(param.b/param.V0)

    ss = StateSpace(As,Bs,Cs,Ds,Aa,Ba,Ca,Da,Eigs,Eiga) #class

    return ss


def plotMotionsTest(param,fileName,t0,duration,StateSpace,motionName,plotNumerical,SI=True):
    '''
    DESCRIPTION:    Plot eigenmotions from flight test data
    ========
    INPUT:\n
    ... fileName [String]:          File name of flight test data, converted to SI units\n
    ... t0 [Value]:                 Starting time from data that needs to be plotted\n
    ... duration [Value]:           The amount of seconds after t0 that need to be plotted\n
    ... motionName [String]:        Name of motion that needs to be plotted, currently supported: 'phugoid','short period','dutch roll'\n

    OUTPUT:\n
    ... None
    '''

    # get data from simulation
    tS, XoutS, YoutS, tA, XoutA, YoutA = calcResponse(t0,duration,fileName,StateSpace,param,SI)

    # get data from dynamic file
    dfTime = imDyn.sliceTime(fileName,t0,duration,SI)
    time = dfTime['time'].to_numpy()

    if motionName=='phugoid' or motionName=='short period':
        # get all variables
        Vt0 = dfTime['Dadc1_tas'].to_numpy()[0]
        Vt = dfTime['Dadc1_tas'].to_numpy()
        ubar =  (Vt-Vt0)/Vt0
        a0 = dfTime['vane_AOA'].to_numpy()[0]
        a_stab = dfTime['vane_AOA'].to_numpy()-a0
        theta0 = a0-dfTime['Ahrs1_Pitch'].to_numpy()[0]     # using theta0=gamma0 (see FD lectures notes page 95) and gamma = alpha-theta
        theta_stab = dfTime['Ahrs1_Pitch'].to_numpy()-theta0
        q = dfTime['Ahrs1_bPitchRate'].to_numpy()
        delta_e = dfTime['delta_e'].to_numpy()

        # creat4e figure
        fig, axs = plt.subplots(2,3,figsize=(16,9),dpi=100)
        plt.suptitle('State response to case: %s'%(motionName),fontsize=16)

        #plot elevator deflection
        gs = axs[0, 2].get_gridspec()
        for ax in axs[:, 2]:
            ax.remove()
        axbig = fig.add_subplot(gs[:, 2])
        axbig.plot(time,delta_e)
        axbig.set_ylabel(r'$\delta_{e}$ [rad]', fontsize=12.0)
        axbig.set_xlabel('time [s]', fontsize=12.0)

        #plot \hat{u}, \alpha, \theta and q versus time
        axs[0, 0].plot(time, ubar,label="Flight data" if plotNumerical==True else None)
        axs[0, 0].set_ylabel(r'$\hat{u}$ [-]', fontsize=12.0)
        axs[0, 0].grid()

        axs[0, 1].plot(time, a_stab,label="Flight data" if plotNumerical==True else None)
        axs[0, 1].set_ylabel(r'$\alpha$ [rad]', fontsize=12.0)
        axs[0, 1].grid()

        axs[1, 0].plot(time, theta_stab,label="Flight data" if plotNumerical==True else None)
        axs[1, 0].set_ylabel(r'$\theta$ [rad]', fontsize=12.0)
        axs[1, 0].set_xlabel('time [s]', fontsize=12.0)
        axs[1, 0].grid()

        axs[1, 1].plot(time, q,label="Flight data" if plotNumerical==True else None)
        axs[1, 1].set_ylabel(r'$q$ [rad/s]', fontsize=12.0)
        axs[1, 1].set_xlabel('time [s]', fontsize=12.0)
        axs[1, 1].grid()

        # set labelsize for ticks
        axs[0, 0].tick_params(axis='both', which='major', labelsize=12)
        axs[0, 0].tick_params(axis='both', which='minor', labelsize=12)
        axs[0, 1].tick_params(axis='both', which='major', labelsize=12)
        axs[0, 1].tick_params(axis='both', which='minor', labelsize=12)
        axs[1, 0].tick_params(axis='both', which='major', labelsize=12)
        axs[1, 0].tick_params(axis='both', which='minor', labelsize=12)
        axs[1, 1].tick_params(axis='both', which='major', labelsize=12)
        axs[1, 1].tick_params(axis='both', which='minor', labelsize=12)

        if plotNumerical==True:
            axs[0, 0].plot(tS, YoutS[:,0], label='Simulation')
            axs[0, 1].plot(tS, YoutS[:,1], label='Simulation')
            axs[1, 0].plot(tS, YoutS[:,2], label='Simulation')
            axs[1, 1].plot(tS, YoutS[:,3]*(param.V0/param.c), label='Simulation')
            axs[0,0].legend()
            axs[0, 1].legend()
            axs[1, 0].legend()
            axs[1, 1].legend()

        # to make sure nothing overlaps
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.show()

    elif motionName=='dutch roll' or motionName=='spiral' or motionName =='aper roll':

        # get all variables
        p = dfTime['Ahrs1_bRollRate'].to_numpy()
        r = dfTime['Ahrs1_bYawRate'].to_numpy()

        # createe figrue
        fig, axs = plt.subplots(3,1,figsize=(16, 9), dpi=100)
        plt.suptitle('State response to case: %s' % (motionName), fontsize=16)

        # plot q and r
        axs[0].plot(time, p,label="Flight data" if plotNumerical==True else None)
        axs[0].set_ylabel(r'$p$ [rad/s]', fontsize=12.0)
        axs[0].grid()

        axs[1].plot(time, r,label="Flight data" if plotNumerical==True else None)
        axs[1].set_ylabel(r'$r$ [rad/s]', fontsize=12.0)
        axs[1].grid()

        axs[2].plot(time,dfTime['delta_a'],label="Aileron")
        axs[2].plot(time,dfTime['delta_r'],label="Rudder")
        axs[2].grid()
        axs[2].legend()


        # set tick size
        axs[0].tick_params(axis='both', which='major', labelsize=12)
        axs[0].tick_params(axis='both', which='minor', labelsize=12)
        axs[1].tick_params(axis='both', which='major', labelsize=12)
        axs[1].tick_params(axis='both', which='minor', labelsize=12)
        axs[2].tick_params(axis='both', which='major', labelsize=12)
        axs[2].tick_params(axis='both', which='minor', labelsize=12)

        if plotNumerical==True:
            axs[0].plot(tA, YoutA[:,2]*(2*param.V0/param.b), label='Simulation')
            axs[1].plot(tA, YoutA[:,3]*(2*param.V0/param.b), label='Simulation')
            axs[0].legend()
            axs[1].legend()


        # to make sure nothing overlaps
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.show()
    return


class StateSpace:
    def __init__(self,As,Bs,Cs,Ds,Aa,Ba,Ca,Da,Eigs,Eiga):
        self.As = As
        self.Bs = Bs
        self.Cs = Cs
        self.Ds = Ds
        self.Aa = Aa
        self.Ba = Ba
        self.Ca = Ca
        self.Da = Da
        self.Eigs = Eigs
        self.Eiga = Eiga


class StartTime:
    def __init__(self,fileName):
        if fileName =='reference':
            self.tPhugoid = 3237   -25      #offset to get stationary conditions
            self.tShortPeriod = 3635     #offset to get stationary conditions
            self.tDutchRoll = 3717
            self.tDutchRollYD = 3767
            self.tAperRoll = 3550
            self.tSpiral = 3920
        elif fileName =='actual':
            #need to be filled in manually after obtaining flight test data
            self.tPhugoid = 3120 + 30
            self.tShortPeriod = 3300 + 52
            self.tDutchRoll = 3480 + 42
            self.tDutchRollYD = 3540
            self.tAperRoll = 3360 + 90
            self.tSpiral = 3780 + 1

class DurationTime:
    def __init__(self,fileName):
        if fileName =='reference':
            self.tPhugoid = 220
            self.tShortPeriod = 10
            self.tDutchRoll = 18
            self.tDutchRollYD = 18
            self.tAperRoll = 18
            self.tSpiral = 18
        elif fileName =='actual':
            #need to be filled in manually after obtaining flight test data
            self.tPhugoid = 170
            self.tShortPeriod = 10
            self.tDutchRoll = 18
            self.tDutchRollYD = 18
            self.tAperRoll = 30
            self.tSpiral = 10


class ParametersOld:
    '''
        DESCRIPTION:    Class containing all constant parameters. To find the constant parameters at a certain time during the dynamic measurements, give inputs to this class. For the static measurement series, the class inputs can be left empty.
        ========
        INPUT:\n
        ... Can be left empty when dealing with the static measurement series \n
        ... fileName [String]:          As default set to 'reference'. This is the name of the CSV file containing all dynamic measurements\n
        ... t0 [Value]:                 As default set to 3000. At what time do you want the stationary flight condition variables?\n

        OUTPUT:\n
        ... ParametersOld [Class]:      Class containing the parameters\n
        '''

    #initial unimproved parameters from appendix C
    def __init__(self, fileName ,t0 ,SI=True):
        if SI==True:
            df = imDyn.sliceTime(fileName,t0,1,SI) # duration is set to 1 second but first element is always taken anyway
        elif SI==False:
            df = imDyn.sliceTime(fileName, t0, 1,SI)  # duration is set to 1 second but first element is always taken anyway
        else:
            raise ValueError("Enter SI = True or SI = False")

        self.t0actual = df['time'].to_numpy()[0]
        # Stationary flight condition
        self.hp0 = df['Dadc1_alt'].to_numpy()[0] # pressure altitude in the stationary flight condition [m]
        self.V0 = df['Dadc1_tas'].to_numpy()[0] # true airspeed in the stationary flight condition [m/sec]
        self.alpha0 = df['vane_AOA'].to_numpy()[0] # angle of attack in the stationary flight condition [rad]
        self.th0 = df['Ahrs1_Pitch'].to_numpy()[0] # pitch angle in the stationary flight condition [rad]
        # Aircraft mass

        dfMass = imWeight.weightMeas(fileName)
        dfMassSliced = dfMass.loc[(dfMass['time'] == self.t0actual)]

        self.m =   dfMassSliced['Weight'].to_numpy()[0]/9.81

        # aerodynamic properties
        self.e = 0.8 # Oswald factor [ ]
        self.CD0 = 0.04 # Zero lift drag coefficient [ ]
        self.CLa = 5.084 # Slope of CL-alpha curve [ ]

        # Longitudinal stability
        self.Cma = -0.5626 # longitudinal stabilty [ ]
        self.Cmde = -1.1642 # elevator effectiveness [ ]

        # Aircraft geometry
        self.S = 30.00  # wing area [m^2]
        self.Sh = 0.2 * self.S  # stabiliser area [m^2]
        self.Sh_S = self.Sh / self.S  # [ ]
        self.lh = 0.71 * 5.968  # tail length [m]
        self.c = 2.0569  # mean aerodynamic cord [m]
        self.lh_c = self.lh / self.c  # [ ]
        self.b = 15.911  # wing span [m]
        self.bh = 5.791  # stabilser span [m]
        self.A = self.b ** 2 / self.S  # wing aspect ratio [ ]
        self.Ah = self.bh ** 2 / self.Sh  # stabilser aspect ratio [ ]
        self.Vh_V = 1  # [ ]self.
        self.ih = -2 * pi / 180  # stabiliser angle of incidence [rad]
        self.xcg = 0.25 * self.c
        self.d = 0.69 # fan diameter engine [m]

        # Constant values concerning atmosphere and gravity
        self.rho0 = 1.2250  # air density at sea level [kg/m^3]
        self.lamb = -0.0065  # temperature gradient in ISA [K/m]
        self.Temp0  = 288.15  # temperature at sea level in ISA [K]
        self.pres0 = 101325 # pressure at sea level in ISA [pa]
        self.R      = 287.05  # specific gas constant [m^2/sec^2K]
        self.g      = 9.81  # [m/sec^2] (gravity constant)
        self.gamma = 1.4 # 

        # air density [kg/m^3]
        self.rho    = self.rho0 * pow( ((1+( self.lamb * self.hp0 / self.Temp0))), (-((self.g / (self.lamb *self.R)) + 1)))
        self.W = self.m * self.g  # [N]       (aircraft weight)

        # Constant values concerning aircraft inertia
        self.muc = self.m / (self.rho * self.S * self.c)
        self.mub = self.m / (self.rho * self.S * self.b)
        self.KX2 = 0.019
        self.KZ2 = 0.042
        self.KXZ = 0.002
        self.KY2 = 1.25 * 1.114

        # Aerodynamic constants
        self.Cmac = 0  # Moment coefficient about the aerodynamic centre [ ]
        self.CNwa = self.CLa  # Wing normal force slope [ ]
        self.CNha = 2 * pi * self.Ah / (self.Ah + 2)  # Stabiliser normal force slope [ ]
        self.depsda = 4 / (self.A + 2)  # Downwash gradient [ ]

        # Lift and drag coefficient
        self.CL = 2 * self.W / (self.rho * self.V0 ** 2 * self.S)  # Lift coefficient [ ]
        self.CD = self.CD0 + (self.CLa * self.alpha0) ** 2 / (pi * self.A * self.e)  # Drag coefficient [ ]

        # Stabiblity derivatives
        self.CX0 = self.W * sin(self.th0) / (0.5 * self.rho * self.V0 ** 2 * self.S)
        self.CXu = -0.095
        self.CXa = +0.47966  # Positive! (has been erroneously negative since 1993)
        self.CXadot = +0.08330
        self.CXq = -0.28170
        self.CXde = -0.03728

        self.CZ0 = -self.W * cos(self.th0) / (0.5 * self.rho * self.V0 ** 2 * self.S)
        self.CZu = -0.37616
        self.CZa = -5.74340
        self.CZadot = -0.00350
        self.CZq = -5.66290
        self.CZde = -0.69612

        self.Cmu = +0.06990
        self.Cmadot = +0.17800
        self.Cmq = -8.79415
        self.CmTc = -0.0064

        self.CYb = -0.7500
        self.CYbdot = 0
        self.CYp = -0.0304
        self.CYr = +0.8495
        self.CYda = -0.0400
        self.CYdr = +0.2300

        self.Clb = -0.10260
        self.Clp = -0.71085
        self.Clr = +0.23760
        self.Clda = -0.23088
        self.Cldr = +0.03440

        self.Cnb = +0.1348
        self.Cnbdot = 0
        self.Cnp = -0.0602
        self.Cnr = -0.2061
        self.Cnda = -0.0120
        self.Cndr = -0.0939


def main():
    inputFile = input("\nChoose to evaluate the 'reference' or 'actual' data: ")
    while inputFile not in ['reference', 'actual']:
        inputFile = input("Invalid input: choose between 'reference' or 'actual'")
    showPlot = input("\nDo you want nice plots? 'Yes' or 'No': ")
    while showPlot not in ['Yes', 'No','yes','no']:
        showPlot = input("Invalid input: choose between 'Yes' or 'No'")
    doSimulate = input("\nDo you want to plot the simulation? 'Yes' or 'No': ")
    while doSimulate not in ['Yes', 'No','yes','no']:
        doSimulate = input("Invalid input: choose between 'Yes' or 'No'")

    if doSimulate == 'Yes' or doSimulate =='yes':
        plotNumerical = True
    else:
        plotNumerical = False


    tStart = StartTime(fileName=inputFile)
    tDuration = DurationTime(fileName=inputFile)

    paramPhugoid = ParametersOld(fileName=inputFile,t0=tStart.tPhugoid,SI=True) #Create parameters for phugoid motion
    paramShortPeriod = ParametersOld(fileName=inputFile,t0=tStart.tShortPeriod,SI=True)
    paramDutchRoll = ParametersOld(fileName=inputFile,t0=tStart.tDutchRoll,SI=True)
    paramDutchRollYD = ParametersOld(fileName=inputFile, t0=tStart.tDutchRollYD,SI=True)
    paramAperRoll = ParametersOld(fileName=inputFile, t0=tStart.tAperRoll,SI=True)
    paramSpiral = ParametersOld(fileName=inputFile, t0=tStart.tSpiral,SI=True)

    ssPhugoid = stateSpace(paramPhugoid)
    ssShortPeriod = stateSpace(paramShortPeriod)
    ssDutchRoll = stateSpace(paramDutchRoll)
    ssDutchRollYD = stateSpace(paramDutchRollYD)
    ssAperRoll = stateSpace(paramAperRoll)
    ssSpiral = stateSpace(paramSpiral)

    eigPhugoid,eigPhugoidSimp = calcEigenPhugoid(paramPhugoid)
    eigShortPeriod = calcEigenShortPeriod(paramShortPeriod)
    eigDutchRoll,eigDutchRollSimp = calcEigenDutchRoll(paramDutchRoll)
    eigDutchRollYD, eigDutchRollYDSimp = calcEigenDutchRoll(paramDutchRoll)
    eigAperRoll = calcEigenAperRoll(paramAperRoll)
    eigSpiral = calcEigenSpiral(paramSpiral)

    print("eigenvalues ss Phugoid: ",ssPhugoid.Eigs[2:])
    print("eigenvalues analytical Phugoid:",eigPhugoid)
    print("eigenvalues analytical Phugoid Simplified:", eigPhugoidSimp,"\n")

    print("eigenvalues ss Short Period: ", ssShortPeriod.Eigs[:2])
    print("eigenvalues analytical Short Period:", eigShortPeriod,"\n")

    print("eigenvalues ss dutch roll:", ssDutchRoll.Eiga[1:3])
    print("eigenvalues analytical dutch roll:", eigDutchRoll)
    print("eigenvalues analytical dutch roll Simplified:",eigDutchRollSimp,"\n")

    print("eigenvalues ss dutch roll YD:", ssDutchRollYD.Eiga[1:3])
    print("eigenvalues analytical dutch roll YD:", eigDutchRollYD)
    print("eigenvalues analytical dutch roll YD Simplified:",eigDutchRollYDSimp,"\n")

    print("eigenvalues ss aperiodic roll",ssAperRoll.Eiga[0])
    print("eigenvalue analytical damped aperiodic roll:",eigAperRoll,"\n")

    print("eigenvalue ss spiral:", ssSpiral.Eiga[3])
    print("eigenvalue analytical spiral:" ,eigSpiral)

    #-----------------------------------------------------
    # plot eigen motions from flight test data or reference data
    #-----------------------------------------------------
    if showPlot =='Yes' or showPlot =='yes':
        plotMotionsTest(paramPhugoid,inputFile,tStart.tPhugoid,tDuration.tPhugoid,ssPhugoid,'phugoid',plotNumerical,SI=True)  # plot from reference data for phugoid
        plotMotionsTest(paramShortPeriod, inputFile, tStart.tShortPeriod, tDuration.tShortPeriod, ssShortPeriod, 'short period', plotNumerical, SI=True)  # plot from reference data for short period
        plotMotionsTest(paramDutchRoll, inputFile, tStart.tDutchRoll, tDuration.tDutchRoll, ssDutchRoll, 'dutch roll',plotNumerical, SI=True)
        plotMotionsTest(paramAperRoll, inputFile, tStart.tAperRoll, tDuration.tAperRoll, ssAperRoll, 'aper roll', plotNumerical,SI=True)
        plotMotionsTest(paramSpiral, inputFile, tStart.tSpiral, tDuration.tSpiral, ssSpiral, 'spiral', plotNumerical,SI=True)

if __name__ == "__main__":
    #this is run when script is started, dont change
    main()

