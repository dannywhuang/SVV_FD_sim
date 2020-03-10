#Initial numerical model before making improvements to parameters
import numpy as np
from math import cos, pi, sin, tan, pow
from import_dynamic import sliceTime
import control.matlab as ml
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

#global constants
g = 9.81

def sampleFunction(param):
    #----------------------------------------------------------------------------------------------
    #
    # Short description of what the function does
    #
    # Input:    name [type]                             description of the variable
    #           name2 [type2]                           description of the variable2
    #           ...                                     ...
    # Output:   name [type]                             description of the variable
    #           name2 [type2]                           description fothe varaible2
    #           ...                                     ...
    #
    #----------------------------------------------------------------------------------------------

    s = param.b+param.S+param.c
    return s


def calcResponse(t0,duration,fileName,param):
    #----------------------------------------------------------------------------------------------
    #
    # Calculate responses using dynamic measurement data
    #
    # Input:    t0 [Value]          The time at which the response starts
    #           duration [Value]    The duration of the response
    #           fileName [String]   Name of dynamic measurement data file, so reference or flighttest
    #           param [Class]       Class with paramaters of aircraft
    #
    # Output:   XoutS [Array]       Array with state variable responses for symmetric
    #           YoutS [Array]       Array with output variable responses for symmetric
    #           XoutA [Array]       Array with state variable responses for asymmetric
    #           YoutS [Array]       Array with output variable responses for asymmetric
    #
    #----------------------------------------------------------------------------------------------

    # get state matrices using aircraft parameters
    As, Bs, Cs, Ds, Aa, Ba, Ca, Da = stateSpace(param)

    # create state space models
    stateSpaceS = ml.ss(As,Bs,Cs,Ds)
    stateSpaceA = ml.ss(Aa,Ba,Ca,Da)

    # get data from the dynamic measurement file
    dfTime = sliceTime(fileName,t0,duration)
    time = dfTime['time']
    delta_a = dfTime['delta_a']
    delta_e = dfTime['delta_e']
    delta_r = dfTime['delta_r']

    # input variables
    us = delta_e
    ua = np.vstack((delta_a,delta_r)).T

    # initial condition
    x0s = [0 , 0 , 0 , dfTime['Ahrs1_bPitchRate'].to_numpy()[0]*(param.c/dfTime['time'].to_numpy()[0])]
    x0a = [0 , dfTime['Ahrs1_Roll'].to_numpy()[0] ,(dfTime['Ahrs1_bPitchRate'].to_numpy()[0]*param.b)/(2*param.V0)  , (dfTime['Ahrs1_bRollRate'].to_numpy()[0]*param.b)/(2*param.V0)]

    # calculate responses
    YoutS, tS, XoutS = ml.lsim(stateSpaceS, us, time, x0s)
    YoutA, tA, XoutA = ml.lsim(stateSpaceA, ua, time, x0a)
    
    return XoutS, YoutS, XoutA, YoutA, tS, tA


def stateSpace(param):
    #----------------------------------------------------------------------------------------------
    #
    # Calculate state-space matrices for symmetric and asymmetric equations of motion
    #
    # Input:    param [Class]                           Class containing aerodynamic and stability parameters, same parameters as in Cit_par.py
    # Output:   As,Bs,Cs,Ds,Aa,Ba,Ca,Da [np.matrix]     State-space matrices A,B,C,D for (s)ymmetric and (a)symmetric equations of moton
    #
    #----------------------------------------------------------------------------------------------

    C1s = np.matrix([[ -2*param.muc*(param.c/param.V0) , 0 , 0 , 0 ],
                     [ 0 , (param.CZadot-2*param.muc)*(param.c/param.V0) , 0 , 0 ],
                     [ 0 , 0 , -(param.c/param.V0) , 0 ],
                     [ 0 , param.Cmadot , 0 , -2*param.muc*param.KY2*(param.c/param.V0)]])

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
                     [ param.Cnbdot , 0 , 4*param.mub*param.KXZ*(param.b/param.V0) , -4*param.mub*param.KZ2*(param.b/param.V0) ]])

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

    return As,Bs,Cs,Ds,Aa,Ba,Ca,Da


def plotMotionsTest(fileName,t0,duration,motionName):
    #----------------------------------------------------------------------------------------------
    #
    # Plot eigenmotions from flight test data
    #
    # Input:    fileName [String]                       File name of flight test data, converted to SI units
    #           t0 [Value]                              Starting time from data that needs to be plotted
    #           duration [Value]                        The amount of seconds after t0 that need to be plotted
    #           motionName [String]                     Name of motion that needs to be plotted, currently supported: 'phugoid','short period','dutch roll'
    # Output:   None
    #
    #----------------------------------------------------------------------------------------------
    if 'SI' not in fileName:
        print('Please use the file name of the flight data in SI units')
        return
    else:
        dfTime = sliceTime(fileName,t0,duration)
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
            axs[0, 0].plot(time, ubar,label=r'$\hat{u}$')
            axs[0, 0].set_ylabel(r'$\hat{u}$ [-]', fontsize=12.0)
            axs[0, 0].grid()
            axs[0,0].legend()

            axs[0, 1].plot(time, a_stab)
            axs[0, 1].set_ylabel(r'$\alpha$ [rad]', fontsize=12.0)
            axs[0, 1].grid()

            axs[1, 0].plot(time, theta_stab)
            axs[1, 0].set_ylabel(r'$\theta$ [rad]', fontsize=12.0)
            axs[1, 0].set_xlabel('time [s]', fontsize=12.0)
            axs[1, 0].grid()

            axs[1, 1].plot(time, q)
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

            # to make sure nothing overlaps
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            plt.show()

        elif motionName=='dutch roll':

            # get all variables
            q = dfTime['Ahrs1_bPitchRate'].to_numpy()
            r = dfTime['Ahrs1_bRollRate'].to_numpy()

            # createe figrue
            fig, axs = plt.subplots(figsize=(16, 9), dpi=100)
            plt.suptitle('State response to case: %s' % (motionName), fontsize=16)

            # plot q and r
            axs.plot(time, q,label='pitch rate')
            axs.plot(time, r,label='roll rate')
            axs.set_ylabel(r'$p,r$ [rad/s]', fontsize=12.0)
            axs.set_xlabel('time [s]', fontsize=12.0)
            axs.grid()
            axs.legend()

            # set tick size
            axs.tick_params(axis='both', which='major', labelsize=12)
            axs.tick_params(axis='both', which='minor', labelsize=12)

            # to make sure nothing overlaps
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            plt.show()
    return


class ParametersOld:
    #initial unimproved parameters from appendix C
    def __init__(self,fileName,t0):
        # ----------------------------------------------------------------------------------------------
        #
        # Short description of what the function does
        #
        # Input:
        #           fileName [String]       If you choose dynamic, what is the fileName of the .csv file? This gives you an option to choose between reference data and actual flight test data
        #           t0 [Value]              At what time do you want the stationary flight condition variables?
        # Output:   ParametersOld [Class]   Class containing the parameters
        #
        # ----------------------------------------------------------------------------------------------

        df = sliceTime(fileName,t0,1) # duration is set to 1 second but first element is always taken anyway

        # Stationary flight condition
        self.hp0 = df['Dadc1_alt'].to_numpy()[0] # pressure altitude in the stationary flight condition [m]
        self.V0 = df['Dadc1_tas'].to_numpy()[0] # true airspeed in the stationary flight condition [m/sec]
        self.alpha0 = df['vane_AOA'].to_numpy()[0] # angle of attack in the stationary flight condition [rad]
        self.th0 = df['Ahrs1_Pitch'].to_numpy()[0] # pitch angle in the stationary flight condition [rad]
        # Aircraft mass
        self.m = 123  # use calculate weight function

        ### CHANGE ABOVE VALUES (123) TO VALUES FROM DYNAMIC MEASUREMENTS

        # Standard values
        self.Ws = 60500 # Standard weight [N]
        self.ms = 0.048 # Standard mass flow [kg/s]

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

        # Constant values concerning atmosphere and gravity
        self.rho0 = 1.2250  # air density at sea level [kg/m^3]
        self.lamb = -0.0065  # temperature gradient in ISA [K/m]
        self.Temp0  = 288.15  # temperature at sea level in ISA [K]
        self.pres0 = 101325 # pressure at sea level in ISA [pa]
        self.R      = 287.05  # specific gas constant [m^2/sec^2K]
        self.g      = 9.81  # [m/sec^2] (gravity constant)
        self.gamma = 1.4 # 

        # air density [kg/m^3]
        # self.rho    = self.rho0 * pow( ((1+( self.lamb * self.hp0 / self.Temp0))), (-((self.g / (self.lamb *self.R)) + 1)))
        # self.W = self.m * self.g  # [N]       (aircraft weight)

        # Constant values concerning aircraft inertia
        # self.muc = self.m / (self.rho * self.S * self.c)
        # self.mub = self.m / (self.rho * self.S * self.b)
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
        # self.CL = 2 * self.W / (self.rho * self.V0 ** 2 * self.S)  # Lift coefficient [ ]
        # self.CD = self.CD0 + (self.CLa * self.alpha0) ** 2 / (pi * self.A * self.e)  # Drag coefficient [ ]

        # Stabiblity derivatives
        # self.CX0 = self.W * sin(self.th0) / (0.5 * self.rho * self.V0 ** 2 * self.S)
        self.CXu = -0.02792
        self.CXa = +0.47966  # Positive! (has been erroneously negative since 1993)
        self.CXadot = +0.08330
        self.CXq = -0.28170
        self.CXde = -0.03728

        # self.CZ0 = -self.W * cos(self.th0) / (0.5 * self.rho * self.V0 ** 2 * self.S)
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
    tPhugoid = 3237
    tShortPeriod = 3635
    tDutchRoll = 3717
    tDutchRollYD = 3767
    tAperRoll = 3550
    tSpiral = 3920
    # create parameters for different motions from data
    paramPhugoid = ParametersOld('reference',tPhugoid) #Create parameters for phugoid motion

    # create state space matrices for phugoid
    As, Bs, Cs, Ds, Aa, Ba, Ca, Da = stateSpace(paramPhugoid)
    print("State matrices are:")
    print("As: ", As)
    print("Bs: ", Bs)
    print("Cs: ", Cs)
    print("Ds: ", Ds)
    print("Aa: ", Aa)
    print("Ba: ", Ba)
    print("Ca: ", Ca)
    print("Da: ", Da)

    # simulate response using relevant parameters
    XoutS, YoutS, XoutA, YoutA, tS, tA = calcResponse(tPhugoid,20,'reference',paramPhugoid)


    #-----------------------------------------------------
    # plot eigen motions from flight test data or reference data
    #-----------------------------------------------------
    #plotMotionsTest('reference_SI',3237,220,'phugoid')  # plot from reference data for phugoid
    #plotMotionsTest('reference_SI',3635,10,'short period')  # plot from reference data for short period
    #plotMotionsTest('reference_SI',3717,18,'dutch roll')  # plot from reference data for dutch roll

# if __name__ == "__main__":
#     #this is run when script is started, dont change
#     main()

