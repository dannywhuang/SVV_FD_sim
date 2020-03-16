import math 


class parametersStatic:
    def __init__(self):
        # Aircraft geometry:
        self.S = 30.00                      # wing area [m^2]
        self.Sh = 0.2 * self.S              # stabiliser area [m^2]
        self.Sh_S = self.Sh / self.S        # [ ]
        self.lh = 0.71 * 5.968              # tail length [m]
        self.c = 2.0569                     # mean aerodynamic cord [m]
        self.lh_c = self.lh / self.c        # [ ]
        self.b = 15.911                     # wing span [m]
        self.bh = 5.791                     # stabilser span [m]
        self.A = self.b ** 2 / self.S       # wing aspect ratio [ ]
        self.Ah = self.bh ** 2 / self.Sh    # stabilser aspect ratio [ ]
        self.Vh_V = 1                       # [ ]self.
        self.ih = -2 * math.pi / 180             # stabiliser angle of incidence [rad]
        self.d = 0.69                       # fan diameter engine [m]

        # Standard values:
        self.Ws = 60500                     # Standard weight [N]
        self.ms = 0.048                     # Standard mass flow [kg/s]

        # Constant values concerning atmosphere and gravity:
        self.rho0 = 1.2250                  # air density at sea level [kg/m^3]
        self.lamb = -0.0065                 # temperature gradient in ISA [K/m]
        self.Temp0  = 288.15                # temperature at sea level in ISA [K]
        self.pres0 = 101325                 # pressure at sea level in ISA [pa]
        self.R      = 287.05                # specific gas constant [m^2/sec^2K]
        self.g      = 9.81                  # [m/sec^2] (gravity constant)
        self.gamma = 1.4                    # 

        # Stability derivatives:
        self.CmTc = -0.0064


class parametersWeight:
    def __init__(self, inputFile):

        if inputFile == 'reference':

            self.mpilot1 = 95                   # [kg]
            self.mpilot2 = 92                   # [kg]
            self.mcoordinator = 74              # [kg]
            self.mobserver1L = 66               # [kg]
            self.mobserver1R = 61               # [kg]
            self.mobserver2L = 75               # [kg]
            self.mobserver2R = 78               # [kg]
            self.mobserver3L = 86               # [kg]
            self.mobserver3R = 68               # [kg]

            self.mblockfuel = 4050*0.45359237   # [kg]

            self.position1 = 288
            self.position2 = 134

            self.locSwitch = 7                  # Initial Seat Location of the person who switched
            self.MBem = 9165*0.45359
            self.MomBem = 2672953.5*0.45359*0.0254

        if inputFile == 'actual':
            # **** Change these values to the correct ones *****
            
            self.mpilot1 = 80                   # [kg]
            self.mpilot2 = 102                  # [kg]
            self.mcoordinator = 60              # [kg]
            self.mobserver1L = 67               # [kg]
            self.mobserver1R = 59               # [kg]
            self.mobserver2L = 78               # [kg]
            self.mobserver2R = 66               # [kg]
            self.mobserver3L = 86               # [kg]
            self.mobserver3R = 87.5             # [kg]

            self.mblockfuel = 3590 * 0.45359237 # [kg]

            # These 5 constants have to be changed to the correct ones.
            self.position1 = 288
            self.position2 = 134

            self.locSwitch = 7                  # Initial Seat Location of the person who switched
            self.MBem = 9165*0.45359
            self.MomBem = 2672953.5*0.45359*0.0254

