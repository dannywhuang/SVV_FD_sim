#Initial numerical model before making improvements to parameters
import numpy as np
from math import cos, pi, sin, tan, pow

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

class ParametersOld:
    #initial unimproved parameters from appendix C
    def __init__(self):

        # Stationary flight condition
        self.hp0 = 123 # pressure altitude in the stationary flight condition [m]
        self.V0 = 123 # true airspeed in the stationary flight condition [m/sec]
        self.alpha0 = 123 # angle of attack in the stationary flight condition [rad]
        self.th0 = 123 # pitch angle in the stationary flight condition [rad]

        # Aircraft mass
        self.m = 123  # mass [kg]

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
        self.rho    = self.rho0 * pow( ((1+( self.lamb * self.hp0 / self.Temp0))), (-((g / (self.lamb *self.R)) + 1)))
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
        self.CXu = -0.02792
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
    #invoking functions should be done in main()
    param = ParametersOld()

    s = sampleFunction(param)
    print("s is: ", s)

    As, Bs, Cs, Ds, Aa, Ba, Ca, Da = stateSpace(param)
    print("State matrices are:")
    print("As: ", As)
    print("Bs: ", Bs)
    print("Cs: ", Cs)
    print("Ds: ", Ds)
    print("Aa: ", Aa)
    print("Ba: ", Ba)
    print("Ca: ", Ca)
    print("Da: ", Da)


    return

if __name__ == "__main__":
    #this is run when script is started, dont change
    main()
