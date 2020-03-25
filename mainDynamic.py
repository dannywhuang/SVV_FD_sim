import numpy as np
from scipy import interpolate, signal
from math import cos, pi, sin, tan, pow
import control.matlab as ml
import matplotlib.pyplot as plt

import import_dynamic as imDyn
import import_weight as imWeight
import staticCalc1 as calc1
import staticCalc2 as calc2


def plot_initial_value_Response(StateSpace, motion):
    '''
    DESCRIPTION:                     Plot initial value responses using dynamic measurement data
    ========
    INPUT:\n
    ... motion [String]:            'symmetric' or 'asymmetric'
    ... Statespace [Class]:         Class with symmetric and asymmetric statespace\n

    OUTPUT:\n
    ... None
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
    if motion== 'symmetric':
         ss = ml.ss(As,Bs,Cs,Ds)
    if motion == 'asymmetric':
         ss = ml.ss(Aa,Ba,Ca,Da)

    # initial value vector
    a = np.matrix('1;0;0;0')
    b = np.matrix('0;1;0;0')
    c = np.matrix('0;0;1;0')
    d = np.matrix('0;0;0;1')
    t = np.arange(0,40,0.001)
    # response
    res1 = (ml.initial(ss,T=t,X0=a))[0][:,0]
    res2 = (ml.initial(ss, T=t, X0=b))[0][:,1]
    res3 = (ml.initial(ss,T=t,X0=c))[0][:,2]
    res4 = (ml.initial(ss, T=t, X0=d))[0][:,3]

    fig, axs = plt.subplots(4, 1, figsize=(16, 9), dpi=100)
    plt.suptitle('Initial value response for ' + (motion) + ' motion', fontsize=16)

    # plot q and r
    axs[0].plot(t, res1)
    axs[0].set_ylabel(r'$\hat{u}$ [-]' if motion== 'symmetric' else r'$\beta$ [rad]', fontsize=14.0)
    axs[0].grid()

    axs[1].plot(t, res2 )
    axs[1].set_ylabel(r'$\alpha$ [rad]' if motion== 'symmetric' else r'$\psi$ [rad]', fontsize=14.0)
    axs[1].grid()

    axs[2].plot(t, res3 )
    axs[2].set_ylabel(r'$\theta$ [rad]'if motion== 'symmetric' else r'p [rad/s]' , fontsize=14.0)
    axs[2].grid()

    axs[3].plot(t, res4)
    axs[3].set_ylabel(r'q [rad/s]'if motion== 'symmetric' else r'q [rad/s]', fontsize=14.0)
    axs[3].set_xlabel(r'time [s]',fontsize=14.0)
    axs[3].grid()


    # set tick size
    axs[0].tick_params(axis='both', which='major', labelsize=13)
    axs[0].tick_params(axis='both', which='minor', labelsize=13)
    axs[1].tick_params(axis='both', which='major', labelsize=13)
    axs[1].tick_params(axis='both', which='minor', labelsize=13)
    axs[2].tick_params(axis='both', which='major', labelsize=13)
    axs[2].tick_params(axis='both', which='minor', labelsize=13)
    axs[3].tick_params(axis='both', which='major', labelsize=13)
    axs[3].tick_params(axis='both', which='minor', labelsize=13)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()


def plotVerificationZeroResponse(StateSpace, motion):
    '''
    DESCRIPTION:                     Plot zero initial value zero input response using dynamic measurement data
    ========
    INPUT:\n
    ... motion [String]:            'symmetric' or 'asymmetric'
    ... Statespace [Class]:         Class with symmetric and asymmetric statespace\n

    OUTPUT:\n
    ... None
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
    t = np.arange(0, 40, 0.001)
    # create state space models
    if motion== 'symmetric':
        ss = ml.ss(As,Bs,Cs,Ds)
        u = np.zeros(len(t))
    if motion == 'asymmetric':
        ss = ml.ss(Aa,Ba,Ca,Da)
        u1 = np.zeros(len(t))
        u2 =np.zeros(len(t))
        u = np.vstack((u1, u2)).T

    # initial value vector
    x0 = np.matrix('0;0;0;0')

    # response
    YoutS, tS, XoutS = ml.lsim(ss, u, t, x0)
    res1 = YoutS[:,0]
    res2 = YoutS[:,1]
    res3 = YoutS[:,2]
    res4 = YoutS[:,3]

    fig, axs = plt.subplots(4, 1, figsize=(16, 9), dpi=100)
    plt.suptitle('Zero input zero initial value response for  ' + (motion) + ' motion', fontsize=16)

    # plot q and r
    axs[0].plot(t, res1)
    axs[0].set_ylabel(r'$\hat{u}$ [-]' if motion== 'symmetric' else r'$\beta$ [rad]', fontsize=14.0)
    axs[0].grid()

    axs[1].plot(t, res2 )
    axs[1].set_ylabel(r'$\alpha$ [rad]' if motion== 'symmetric' else r'$\psi$ [rad]', fontsize=14.0)
    axs[1].grid()

    axs[2].plot(t, res3 )
    axs[2].set_ylabel(r'$\theta$ [rad]'if motion== 'symmetric' else r'p [rad/s]' , fontsize=14.0)
    axs[2].grid()

    axs[3].plot(t, res4)
    axs[3].set_ylabel(r'q [rad/s]'if motion== 'symmetric' else r'q [rad/s]', fontsize=14.0)
    axs[3].grid()


    # set tick size
    axs[0].tick_params(axis='both', which='major', labelsize=13)
    axs[0].tick_params(axis='both', which='minor', labelsize=13)
    axs[1].tick_params(axis='both', which='major', labelsize=13)
    axs[1].tick_params(axis='both', which='minor', labelsize=13)
    axs[2].tick_params(axis='both', which='major', labelsize=13)
    axs[2].tick_params(axis='both', which='minor', labelsize=13)
    axs[3].tick_params(axis='both', which='major', labelsize=13)
    axs[3].tick_params(axis='both', which='minor', labelsize=13)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()
    return


def plotVerificationStepResponse(StateSpace, motion):
    '''
    DESCRIPTION:                     Plot zero initial value step input response for dynamic measurement data
    ========
    INPUT:\n
    ... motion [String]:            'symmetric' or 'asymmetric'
    ... Statespace [Class]:         Class with symmetric and asymmetric statespace\n

    OUTPUT:\n
    ... None
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
    t = np.arange(0, 400, 0.1)
    # create state space models
    if motion== 'symmetric':
        ss = ml.ss(As,Bs,Cs,Ds)
        u = np.ones(len(t))
    if motion == 'asymmetric':
        ss = ml.ss(Aa,Ba,Ca,Da)
        u1 = np.ones(len(t))
        u2 =np.ones(len(t))
        u = np.vstack((u1, u2)).T

    # initial value vector
    x0 = np.matrix('0;0;0;0')

    # response
    YoutS, tS, XoutS = ml.lsim(ss, u, t, x0)
    res1 = YoutS[:,0]
    res2 = YoutS[:,1]
    res3 = YoutS[:,2]
    res4 = YoutS[:,3]

    fig, axs = plt.subplots(4, 1, figsize=(16, 9), dpi=100)
    plt.suptitle('Step input zero initial value response for  ' + (motion) + ' motion', fontsize=16)

    # plot q and r
    axs[0].plot(t, res1)
    axs[0].set_ylabel(r'$\hat{u}$ [-]' if motion== 'symmetric' else r'$\beta$ [rad]', fontsize=14.0)
    axs[0].grid()

    axs[1].plot(t, res2 )
    axs[1].set_ylabel(r'$\alpha$ [rad]' if motion== 'symmetric' else r'$\psi$ [rad]', fontsize=14.0)
    axs[1].grid()

    axs[2].plot(t, res3 )
    axs[2].set_ylabel(r'$\theta$ [rad]'if motion== 'symmetric' else r'p [rad/s]' , fontsize=14.0)
    axs[2].grid()

    axs[3].plot(t, res4)
    axs[3].set_ylabel(r'q [rad/s]'if motion== 'symmetric' else r'q [rad/s]', fontsize=14.0)
    axs[3].grid()


    # set tick size
    axs[0].tick_params(axis='both', which='major', labelsize=13)
    axs[0].tick_params(axis='both', which='minor', labelsize=13)
    axs[1].tick_params(axis='both', which='major', labelsize=13)
    axs[1].tick_params(axis='both', which='minor', labelsize=13)
    axs[2].tick_params(axis='both', which='major', labelsize=13)
    axs[2].tick_params(axis='both', which='minor', labelsize=13)
    axs[3].tick_params(axis='both', which='major', labelsize=13)
    axs[3].tick_params(axis='both', which='minor', labelsize=13)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()
    return


def plotVerificationImpulseResponse(StateSpace, motion):
    '''
    DESCRIPTION:                     Plot zero initial value impulse input response for dynamic measurement data
    ========
    INPUT:\n
    ... motion [String]:            'symmetric' or 'asymmetric'
    ... Statespace [Class]:         Class with symmetric and asymmetric statespace\n

    OUTPUT:\n
    ... None
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
    t = np.arange(0, 400, 0.001)
    # create state space models
    if motion== 'symmetric':
        ss = ml.ss(As,Bs,Cs,Ds)
    if motion == 'asymmetric':
        ss = ml.ss(Aa,Ba,Ca,Da)

    # initial value vector
    x0 = np.matrix('0;0;0;0')

    # response
    YoutS, tS = ml.impulse(ss, T=t, X0=x0)
    res1 = YoutS[:,0]
    res2 = YoutS[:,1]
    res3 = YoutS[:,2]
    res4 = YoutS[:,3]

    fig, axs = plt.subplots(4, 1, figsize=(16, 9), dpi=100)
    plt.suptitle('Step input zero initial value response for  ' + (motion) + ' motion', fontsize=16)

    # plot q and r
    axs[0].plot(t, res1)
    axs[0].set_ylabel(r'$\hat{u}$ [-]' if motion== 'symmetric' else r'$\beta$ [rad]', fontsize=14.0)
    axs[0].grid()

    axs[1].plot(t, res2 )
    axs[1].set_ylabel(r'$\alpha$ [rad]' if motion== 'symmetric' else r'$\psi$ [rad]', fontsize=14.0)
    axs[1].grid()

    axs[2].plot(t, res3 )
    axs[2].set_ylabel(r'$\theta$ [rad]'if motion== 'symmetric' else r'p [rad/s]' , fontsize=14.0)
    axs[2].grid()

    axs[3].plot(t, res4)
    axs[3].set_ylabel(r'q [rad/s]'if motion== 'symmetric' else r'q [rad/s]', fontsize=14.0)
    axs[3].grid()


    # set tick size
    axs[0].tick_params(axis='both', which='major', labelsize=13)
    axs[0].tick_params(axis='both', which='minor', labelsize=13)
    axs[1].tick_params(axis='both', which='major', labelsize=13)
    axs[1].tick_params(axis='both', which='minor', labelsize=13)
    axs[2].tick_params(axis='both', which='major', labelsize=13)
    axs[2].tick_params(axis='both', which='minor', labelsize=13)
    axs[3].tick_params(axis='both', which='major', labelsize=13)
    axs[3].tick_params(axis='both', which='minor', labelsize=13)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()
    return


def eigen_dyn_dutchroll(var, param):  # enter 'p' or 'r'
    '''
      DESCRIPTION:                  Calculates eigen values for flight test data for dutch roll
      ========
      INPUT:\n
      ... var [String]:             State variable, nter 'p' or 'r'\n
      ... param [Class]:            Parameters for the eigenmotion\n

      OUTPUT:\n
      ... re [Value]:               Real part eigenvalue\n
      ... im [Value]:               Imaginary part eigenvalue
      '''
    pa = param
    dfTime = imDyn.sliceTime('actual', 3480 + 43, 13, SI=True)
    time = dfTime['time'].to_numpy()

    p = dfTime['Ahrs1_bRollRate'].to_numpy()
    r = dfTime['Ahrs1_bYawRate'].to_numpy()

    if var == 'p':

        p = p

        f = interpolate.UnivariateSpline(list(time), list(p), s=0)

        pp = (f.roots()[4] - f.roots()[2])
        imag = 2 * np.pi * pa.b / (pa.V0 * pp)

        peakind = signal.find_peaks_cwt(list(p), np.arange(1, 10))
        time_peaks, y_peaks = (time[peakind[1:4]]), (p[peakind[1:4]])
        # fp = np.poly1d(np.polyfit(time_peaks, y_peaks, 2))

        a = np.polyfit(time_peaks, np.log(y_peaks), 1)
        fl = lambda x: np.exp(a[1]) * np.exp(a[0] * x)

        time = [i for i in time if i > 3480 + 42 and i < 3480 + 42 + 18]

        half_a0 = (fl(np.array(time))[0]) / 2
        for i in fl(np.array(time)):
            if half_a0 - 0.003 <= i <= half_a0 + 0.003:
                T_ha = time[list(fl(np.array(time))).index(i)]
        re = np.log(0.5) * pa.b / (pa.V0 * (T_ha - time[0]))

        return re, imag

    if var == 'r':
        p = r
        f = interpolate.UnivariateSpline(list(time), list(p), s=0)

        pp = (f.roots()[4] - f.roots()[2])  # 2-0
        imag = 2 * np.pi * pa.b / (pa.V0 * pp)

        peakind = signal.find_peaks_cwt(list(p), np.arange(1, 10))
        time_peaks, y_peaks = (time[peakind[2:]]), (p[peakind[2:]])
        # fp = np.poly1d(np.polyfit(time_peaks, y_peaks, 2))

        a = np.polyfit(time_peaks, np.log(y_peaks), 1)
        fl = lambda x: np.exp(a[1]) * np.exp(a[0] * x)

        time = [i for i in time if i > 3480 + 42 and i < 3480 + 42 + 18]

        half_a0 = (fl(np.array(time))[0]) / 2

        for i in fl(np.array(time)):
            if half_a0 - 0.00325 <= i <= half_a0 + 0.00325:
                T_ha = time[list(fl(np.array(time))).index(i)]
        re = np.log(0.5) * pa.b / (pa.V0 * (T_ha - time[0]))

        return re, imag


def eigen_dyn_phugoid(var, param):
    '''
      DESCRIPTION:                  Calculates eigen values for flight test data for phugoid
      ========
      INPUT:\n
      ... var [String]:             State variable, enter 'q' , 'ubar' , 'theta_stab'\n
      ... param [Class]:            Parameters for the eigenmotion\n

      OUTPUT:\n
      ... re [Value]:               Real part eigenvalue\n
      ... im [Value]:               Imaginary part eigenvalue
      '''
    pa = param

    dfTime = imDyn.sliceTime('actual', 3120 + 30, 165, SI=True)
    time = dfTime['time'].to_numpy()

    Vt0 = dfTime['Dadc1_tas'].to_numpy()[0]
    Vt = dfTime['Dadc1_tas'].to_numpy()
    ubar = (Vt - Vt0) / Vt0

    a0 = dfTime['vane_AOA'].to_numpy()[0]
    a_stab = dfTime['vane_AOA'].to_numpy() - a0

    theta0 = dfTime['Ahrs1_Pitch'].to_numpy()[0]  # using theta0=gamma0 (see FD lectures notes page 95) and gamma = alpha-theta
    theta_stab = dfTime['Ahrs1_Pitch'].to_numpy() - theta0

    q = dfTime['Ahrs1_bPitchRate'].to_numpy()

    if var == 'q':
        p = q
        f = interpolate.UnivariateSpline(list(time), list(p), s=0)

        pp = (f.roots()[11] - f.roots()[9])
        imag = 2 * np.pi * pa.c / (pa.V0 * pp)

        peakind, _ = signal.find_peaks(p, width=40)
        time_peaks, y_peaks = (time[peakind[1:]]), (p[peakind[1:]])

        # fp = np.poly1d(np.polyfit(time_peaks, y_peaks, 2))

        a = np.polyfit(time_peaks, np.log(y_peaks), 1)
        fl = lambda x: np.exp(a[1]) * np.exp(a[0] * x)

        time = [i for i in time if i > 3120 + 30 and i < 3432]

        half_a0 = (fl(np.array(time))[0]) / 2
        for i in fl(np.array(time)):
            if half_a0 - 0.000004 <= i <= half_a0 + 0.000004:
                T_ha = time[list(fl(np.array(time))).index(i)]
        re = np.log(0.5) * pa.c / (pa.V0 * (T_ha - time[0]))

        return re, imag

    if var == 'ubar':
        p = -ubar
        f = interpolate.UnivariateSpline(list(time), list(p), s=0)

        pp = (f.roots()[4] - f.roots()[2])
        imag = 2 * np.pi * pa.c / (pa.V0 * pp)

        peakind, _ = signal.find_peaks(p, width=40)
        time_peaks, y_peaks = (time[peakind[1:]]), (p[peakind[1:]])

        # fp = np.poly1d(np.polyfit(time_peaks, y_peaks, 2))

        a = np.polyfit(time_peaks, np.log(y_peaks), 1)
        fl = lambda x: np.exp(a[1]) * np.exp(a[0] * x)

        half_a0 = (fl(np.array(time))[0]) / 2

        for i in fl(np.array(time)):
            if half_a0 - 0.00001 <= i <= half_a0 + 0.00001:
                T_ha = time[list(fl(np.array(time))).index(i)]
        re = np.log(0.5) * pa.c / (pa.V0 * (T_ha - time[0]))

        return re, imag

    if var == 'theta_stab':
        p = theta_stab
        f = interpolate.UnivariateSpline(list(time), list(p), s=0)
        pp = (f.roots()[4] - f.roots()[2])
        imag = 2 * np.pi * pa.c / (pa.V0 * pp)

        peakind, _ = signal.find_peaks(p, width=40)
        time_peaks, y_peaks = (time[peakind[1:]]), (p[peakind[1:]])

        # fp = np.poly1d(np.polyfit(time_peaks, y_peaks, 2))

        a = np.polyfit(time_peaks, np.log(y_peaks), 1)
        fl = lambda x: np.exp(a[1]) * np.exp(a[0] * x)

        half_a0 = (fl(np.array(time))[0]) / 2

        for i in fl(np.array(time)):
            if half_a0 - 0.00005 <= i <= half_a0 + 0.00005:
                T_ha = time[list(fl(np.array(time))).index(i)]
        re = np.log(0.5) * pa.c / (pa.V0 * (T_ha - time[0]))

        return re, imag


def calcDampFrequency(eig,param,type):
    '''
      DESCRIPTION:               Calculates halving time, period, damping coefficient, natural frequency and damped natural frequency
      ========
      INPUT:\n
      ... eig [Value]:          Eigenvalue\n
      ... param [Class]:        Parameters for the eigenmotion\n
      ... type [String]:        Symmetric or asymmetric\n

      OUTPUT:\n
      ... Thalf [Value]:        Damping coefficient\n
      ... P [Value]:            Period\n
      ... zeta [Value]:         Damping ratio\n
      ... w0 [Value]:           Natural frequency\n
      ... wn [Value]:           Damped natural frequency
      '''

    re = np.real(eig)
    im = np.imag(eig)

    if type=='symmetric':
        Thalf = ((np.log(0.5))/re)*(param.c / param.V0)
        if im==0:
            P = None
        else:
            P = ((2 * np.pi) / im) * (param.c / param.V0)
        zeta = -re / (np.sqrt(re**2 + im**2))
        w0 = (np.sqrt(re**2 + im**2)) * (param.V0 / param.b)
        wn = w0 * np.sqrt(1 - zeta**2)
    elif type=='asymmetric':
        Thalf = ((np.log(0.5)) / re) * (param.b / param.V0)
        if im==0:
            P = None
        else:
            P = ((2 * np.pi) / im) * (param.b / param.V0)
        zeta = -re / (np.sqrt(re**2 + im**2))
        w0 = (np.sqrt(re**2 + im**2)) * (param.V0 / param.c)
        wn = w0 * np.sqrt(1 - zeta**2)
    else:
        raise ValueError("Choose symmetric or asymmetric as type")

    return Thalf,P,zeta, w0, wn


def calcEigenShortPeriod(param):
    '''
    DESCRIPTION:                    Calculate eigenvalues of phugoid
    ========
    INPUT:\n
    ... param [Class]:              Class containing aerodynamic and stability parameters, same parameters as in Cit_par.py\n

    OUTPUT:\n
    ... EigLst [Array]:             List containing EigenValue class
    '''

    A = 2*param.muc*param.KY2*(2*param.muc-param.CZadot)
    B = -2*param.muc*param.KY2*param.CZa-(2*param.muc+param.CZq)*param.Cmadot-(2*param.muc-param.CZadot)*param.Cmq
    C = param.CZa*param.Cmq-(2*param.muc+param.CZq)*param.Cma
    p = [A,B,C]
    lambdac = np.roots(p)

    EigLst = []
    for egs in lambdac:
        EigLst.append(EigenValue(egs,param,'symmetric'))

    return EigLst


def calcEigenPhugoid(param):
    '''
    DESCRIPTION:                    Calculate eigenvalues of phugoid
    ========
    INPUT:\n
    ... param [Class]:              Class containing aerodynamic and stability parameters, same parameters as in Cit_par.py\n

    OUTPUT:\n
    ... EigLst [Array]:             List containing EigenValue class\n
    ... EigSimpLst [Array]:         List containing simplified EigenValue class
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

    EigLst = []
    EigSimpLst = []
    for egs in lambdac:
        EigLst.append(EigenValue(egs, param, 'symmetric'))

    for egs2 in lambdacSimp:
        EigSimpLst.append(EigenValue(egs2, param, 'symmetric'))

    return EigLst, EigSimpLst


def calcEigenDutchRoll(param):
    '''
    DESCRIPTION:                    Calculate eigenvalues of dutch roll
    ========
    INPUT:\n
    ... param [Class]:              Class containing aerodynamic and stability parameters, same parameters as in Cit_par.py\n

    OUTPUT:\n
    ... EigLst [Array]:             List containing EigenValue class\n
    ... EigSimpLst [Array]:         List containing simplified EigenValue class
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

    EigLst = []
    EigSimpLst = []
    for egs in lambdac:
        EigLst.append(EigenValue(egs, param, 'asymmetric'))

    for egs2 in lambdacSimp:
        EigSimpLst.append(EigenValue(egs2, param, 'asymmetric'))

    return EigLst, EigSimpLst


def calcEigenAperRoll(param):
    '''
    DESCRIPTION:                Calculate eigenvalues of aperiodic roll
    ========
    INPUT:\n
    ... param [Class]:          Class containing aerodynamic and stability parameters, same parameters as in Cit_par.py\n

    OUTPUT:\n
    ... Eig [Class]:            Class containing eigenvalue, halving time, period, damping factor, natural frequency and damped natural frequency.
    '''

    lambdac = param.Clp/(4*param.mub*param.KX2)

    Eig = EigenValue(lambdac, param, 'asymmetric')

    return Eig


def calcEigenSpiral(param): # Verified by Danny
    '''
    DESCRIPTION:                Calculate eigenvalues of spiral
    ========
    INPUT:\n
    ... param [Class]:          Class containing aerodynamic and stability parameters, same parameters as in Cit_par.py\n

    OUTPUT:\n
    ... Eig [Class]:            Class containing eigenvalue, halving time, period, damping factor, natural frequency and damped natural frequency.
    '''

    lambdac = ((2*param.CL*(param.Clb*param.Cnr-param.Cnb*param.Clr))/(param.Clp*(param.CYb*param.Cnr+4*param.mub*param.Cnb)-param.Cnp*(param.CYb*param.Clr+4*param.mub*param.Clb)))
    Eig = EigenValue(lambdac, param, 'asymmetric')

    return Eig


def calcResponse(t0,duration,fileName,StateSpace,param,SI=True):
    '''
    DESCRIPTION:    Calculate responses using dynamic measurement data
    ========
    INPUT:\n
    ... t0 [Value]:                 The time at which the response starts\n
    ... duration [Value]:           The duration of the response\n
    ... fileName [String]:          Name of dynamic measurement data file, so reference or flighttest\n
    ... StateSpace [Class]:         Class containing state space matrices and EigenValue class\n
    ... param [Class]:              Class with paramaters of aircraft\n
    ... SI [Boolean]:               Set to True if SI units are to be used\n

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
    delta_a = dfTime['delta_a'].to_numpy()-dfTime['delta_a'].to_numpy()[0]
    delta_e = dfTime['delta_e'].to_numpy()-dfTime['delta_e'].to_numpy()[0]
    delta_r = dfTime['delta_r'].to_numpy()-dfTime['delta_r'].to_numpy()[0]

    # input variables
    us = delta_e
    ua = np.vstack((-delta_a,-delta_r)).T

    a0 = dfTime['vane_AOA'].to_numpy()[0]
    a_stab0 = dfTime['vane_AOA'].to_numpy()[0] - a0

    theta0 = dfTime['Ahrs1_Pitch'].to_numpy()[0]
    theta_stab0 = dfTime['Ahrs1_Pitch'].to_numpy()[0]-theta0
    q0 = dfTime['Ahrs1_bPitchRate'].to_numpy()[0]*(param.c/param.V0)

    roll0 = dfTime['Ahrs1_Roll'].to_numpy()[0]

    # initial condition
    x0s = [0 , a_stab0 , theta_stab0 , q0]
    # x0a = [0 , dfTime['Ahrs1_Roll'].to_numpy()[0] ,(dfTime['Ahrs1_bRollRate'].to_numpy()[0]*param.b)/(2*param.V0)  , (dfTime['Ahrs1_bYawRate'].to_numpy()[0]*param.b)/(2*param.V0)]
    x0a = [0 , roll0 , (dfTime['Ahrs1_bRollRate'].to_numpy()[0]*param.b)/(2*param.V0)  , (dfTime['Ahrs1_bYawRate'].to_numpy()[0]*param.b)/(2*param.V0)]
    # calculate responses
    YoutS, tS, XoutS = ml.lsim(stateSpaceS, us, time, x0s)
    YoutA, tA, XoutA = ml.lsim(stateSpaceA, ua, time, x0a)
    
    return tS, XoutS, YoutS, tA, XoutA, YoutA


def stateSpace(param):
    '''
    DESCRIPTION:    Calculate state-space matrices for symmetric and asymmetric equations of motion
    ========
    INPUT:\n
    ... param [Class]:                          Class containing aerodynamic and stability parameters, same parameters as in Cit_par.py\n
    
    OUTPUT:\n
    ... StateSpace [Class]:                     Class containing state-space matrices A,B,C,D and EigenValue classes for (s)ymmetric and (a)symmetric equations of moton
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

    EigsLst = []
    EigaLst = []
    for egs in Eigs:
        EigsLst.append(EigenValue(egs,param,'symmetric'))
    for ega in Eiga:
        EigaLst.append(EigenValue(ega,param,'asymmetric'))

    ss = StateSpace(As,Bs,Cs,Ds,Aa,Ba,Ca,Da,EigsLst,EigaLst) #class

    return ss


def plotMotionsTest(param,fileName,t0,duration,StateSpace,motionName,plotNumerical,SI=True):
    '''
    DESCRIPTION:    Plot eigenmotions from flight test data
    ========
    INPUT:\n
    ... fileName [String]:          File name of flight test data, converted to SI units\n
    ... t0 [Value]:                 Starting time from data that needs to be plotted\n
    ... duration [Value]:           The amount of seconds after t0 that need to be plotted\n
    ... StateSpace [Class]:         Class containing state space matrices and EigenValue class\n
    ... motionName [String]:        Name of motion that needs to be plotted, currently supported: 'phugoid','short period','dutch roll'\n
    ... plotNumerical [Boolean]:    Set to True if the numerical simulation has to be run and plotted\n
    ... SI [Boolean]:               Set to True if SI units are to be used\n

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
        theta0 = dfTime['Ahrs1_Pitch'].to_numpy()[0]     # using theta0=gamma0 (see FD lectures notes page 95) and gamma = alpha-theta
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
        axbig.plot(time,delta_e-delta_e[0],color="green")
        axbig.grid()
        axbig.tick_params(axis='both', which='major', labelsize=13)
        axbig.tick_params(axis='both', which='minor', labelsize=13)
        axbig.set_ylabel(r'$\delta_{e}$ [rad]', fontsize=14.0)
        axbig.set_xlabel('time [s]', fontsize=14.0)

        #plot \hat{u}, \alpha, \theta and q versus time
        axs[0, 0].plot(time, ubar,label="Flight data" if plotNumerical==True else None)
        axs[0, 0].set_ylabel(r'$\hat{u}$ [-]', fontsize=14.0)
        axs[0, 0].grid()

        axs[0, 1].plot(time, a_stab,label="Flight data" if plotNumerical==True else None)
        axs[0, 1].set_ylabel(r'$\alpha$ [rad]', fontsize=14.0)
        axs[0, 1].grid()

        axs[1, 0].plot(time, theta_stab,label="Flight data" if plotNumerical==True else None)
        axs[1, 0].set_ylabel(r'$\theta$ [rad]', fontsize=14.0)
        axs[1, 0].set_xlabel('time [s]', fontsize=14.0)
        axs[1, 0].grid()

        axs[1, 1].plot(time, q,label="Flight data" if plotNumerical==True else None)
        axs[1, 1].set_ylabel(r'$q$ [rad/s]', fontsize=14.0)
        axs[1, 1].set_xlabel('time [s]', fontsize=14.0)
        axs[1, 1].grid()

        # set labelsize for ticks
        axs[0, 0].tick_params(axis='both', which='major', labelsize=13)
        axs[0, 0].tick_params(axis='both', which='minor', labelsize=13)
        axs[0, 1].tick_params(axis='both', which='major', labelsize=13)
        axs[0, 1].tick_params(axis='both', which='minor', labelsize=13)
        axs[1, 0].tick_params(axis='both', which='major', labelsize=13)
        axs[1, 0].tick_params(axis='both', which='minor', labelsize=13)
        axs[1, 1].tick_params(axis='both', which='major', labelsize=13)
        axs[1, 1].tick_params(axis='both', which='minor', labelsize=13)

        if plotNumerical==True:
            axs[0, 0].plot(tS, YoutS[:,0], label='Simulation')
            axs[0, 1].plot(tS, YoutS[:,1], label='Simulation')
            axs[1, 0].plot(tS, YoutS[:,2], label='Simulation')
            axs[1, 1].plot(tS, YoutS[:,3]*(param.V0/param.c), label='Simulation')
            axs[0,0].legend(fontsize=14)
            axs[0, 1].legend(fontsize=14)
            axs[1, 0].legend(fontsize=14)
            axs[1, 1].legend(fontsize=14)

        # to make sure nothing overlaps
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.show()

    elif motionName=='dutch roll' or motionName=='dutch roll yd' or motionName=='spiral' or motionName =='aper roll':

        # get all variables
        p = dfTime['Ahrs1_bRollRate'].to_numpy()
        r = dfTime['Ahrs1_bYawRate'].to_numpy()
        roll = dfTime['Ahrs1_Roll'].to_numpy()
        # createe figrue
        fig, axs = plt.subplots(4,1,figsize=(16, 9), dpi=100)
        plt.suptitle('State response to case: %s' % (motionName), fontsize=16)

        # plot q and r
        axs[0].plot(time, roll, label="Flight data" if plotNumerical == True else None)
        axs[0].set_ylabel(r'$\phi$ [rad]', fontsize=14.0)
        axs[0].grid()

        axs[1].plot(time, p,label="Flight data" if plotNumerical==True else None)
        axs[1].set_ylabel(r'$p$ [rad/s]', fontsize=14.0)
        axs[1].grid()

        axs[2].plot(time, r,label="Flight data" if plotNumerical==True else None)
        axs[2].set_ylabel(r'$r$ [rad/s]', fontsize=14.0)
        axs[2].grid()

        axs[3].plot(time,-(dfTime['delta_a'].to_numpy()-dfTime['delta_a'].to_numpy()[0]),label="Aileron",color="red")
        axs[3].plot(time,-(dfTime['delta_r'].to_numpy()-dfTime['delta_r'].to_numpy()[0]),label="Rudder",color="indigo")
        axs[3].set_ylabel(r'$\delta_{a},\delta_{r}$ [rad]', fontsize=14.0)
        axs[3].set_xlabel('time [s]', fontsize=14.0)
        axs[3].grid()
        axs[3].legend()


        # set tick size
        axs[0].tick_params(axis='both', which='major', labelsize=13)
        axs[0].tick_params(axis='both', which='minor', labelsize=13)
        axs[1].tick_params(axis='both', which='major', labelsize=13)
        axs[1].tick_params(axis='both', which='minor', labelsize=13)
        axs[2].tick_params(axis='both', which='major', labelsize=13)
        axs[2].tick_params(axis='both', which='minor', labelsize=13)
        axs[3].tick_params(axis='both', which='major', labelsize=13)
        axs[3].tick_params(axis='both', which='minor', labelsize=13)

        if plotNumerical==True:
            axs[0].plot(tA,YoutA[:,1],label='Simulation')
            axs[0].legend()
            axs[1].plot(tA, YoutA[:,2]*(2*param.V0/param.b), label='Simulation')
            axs[2].plot(tA, YoutA[:,3]*(2*param.V0/param.b), label='Simulation')
            axs[1].legend()
            axs[2].legend()
        # to make sure nothing overlaps
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.show()
    return


class EigenValue:
    '''
        DESCRIPTION:                    Class containing eigenvalue, halving time, period, damping factor, natural frequency and damped natural frequency.
        ========
        INPUT:\n
        ... fileName [String]:          This is the name of the CSV file containing all dynamic measurements, choose between reference and actual \n

        OUTPUT:\n
        ... EigenValue [Class]:         Class containing eigenvalue, halving time, period, damping factor, natural frequency and damped natural frequency.
        '''
    def __init__(self,eig,param,type):
        self.eig = eig
        self.Thalf = calcDampFrequency(eig,param,type)[0]
        self.P = calcDampFrequency(eig,param,type)[1]
        self.zeta =  calcDampFrequency(eig,param,type)[2]
        self.w0 =  calcDampFrequency(eig,param,type)[3]
        self.wn =  calcDampFrequency(eig,param,type)[4]


class StateSpace:
    '''
        DESCRIPTION:                    Class containing state space matrices and EigenValue class
        ========
        INPUT:\n
        ... fileName [String]:          This is the name of the CSV file containing all dynamic measurements, choose between reference and actual \n

        OUTPUT:\n
        ... StateSpace [Class]:         Class containing state space matrices and EigenValue class
        '''
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
    '''
        DESCRIPTION:                Class containing eigen motion starting times
        ========
        INPUT:\n
        ... fileName [String]:      This is the name of the CSV file containing all dynamic measurements, choose between reference and actual \n

        OUTPUT:\n
        ... StartTime [Class]:      Class containing the starting times
        '''
    def __init__(self,fileName):
        if fileName =='reference':
            self.tPhugoid = 3237   -9      #starts at 3228 ends at 3450
            self.tShortPeriod = 3635 -2     #starts at 3633 ends at 3645
            self.tDutchRoll = 3717 - 1  # starts at 3716 ends at 3735
            self.tDutchRollYD = 3767 -1   # starts at 3766 ends at 3776
            self.tAperRoll = 3550 # starts at 3550 ends at 3562
            self.tSpiral = 3922      # starts at 3922 ends at 4122
        elif fileName =='actual':
            #need to be filled in manually after obtaining flight test data
            self.tPhugoid = 3165  - 15  #starts at 3150 and ends at 3315
            self.tShortPeriod = 3353 -1 #starts at 3352 and ends at 3364
            self.tDutchRoll = 3522 + 1 #starts at 3523 ends at 3536
            self.tDutchRollYD = 3573 # starts at 3573 ends at 3586
            self.tAperRoll = 3403 + 47 # starts at 3450 ends at 3468
            self.tSpiral = 3795  -14 # starts at 3781 ends at 3870      # but maybe stop simulation earlier since it diverges quick


class DurationTime:
    '''
        DESCRIPTION:    Class containing eigen motion duration times
        ========
        INPUT:\n
        ... fileName [String]:          This is the name of the CSV file containing all dynamic measurements, choose between reference and actual \n

        OUTPUT:\n
        ... DurationTime [Class]:      Class containing the duration times
        '''
    def __init__(self,fileName):
        if fileName =='reference':
            self.tPhugoid = 222
            self.tShortPeriod = 12
            self.tDutchRoll = 19
            self.tDutchRollYD = 10
            self.tAperRoll = 12
            self.tSpiral = 200
        elif fileName =='actual':
            self.tPhugoid = 165
            self.tShortPeriod = 12
            self.tDutchRoll = 13
            self.tDutchRollYD = 13
            self.tAperRoll = 18
            self.tSpiral = 89


class ParametersOld:
    '''
        DESCRIPTION:    Class containing all unmatched parameters. To find the constant parameters at a certain time during the dynamic measurements, give inputs to this class. For the static measurement series, the class inputs can be left empty.
        ========
        INPUT:\n
        ... Can be left empty when dealing with the static measurement series \n
        ... fileName [String]:          This is the name of the CSV file containing all dynamic measurements, choose between reference and actual \n
        ... t0 [Value]:                 At what time do you want the stationary flight condition variables? \n
        ... SI [Boolean]:               Set to True for SI units, False for non-SI units \n

        OUTPUT:\n
        ... ParametersOld [Class]:      Class containing the parameters\n
        '''

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
        aeroCoeff = calc1.calcAeroCoeff(fileName)
        self.e = aeroCoeff[2] # Oswald factor [ ]
        self.CD0 = aeroCoeff[3] # Zero lift drag coefficient [ ]
        self.CLa = aeroCoeff[0] # Slope of CL-alpha curve [ ]

        # Longitudinal stability
        self.Cmde = calc2.calcElevEffectiveness(fileName)
        self.Cma     = calc2.calcElevDeflection(fileName)[1]

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


class ParametersAsymmetric:
    '''
        DESCRIPTION:    Class containing all asymmetric matched parameters. To find the constant parameters at a certain time during the dynamic measurements, give inputs to this class. For the static measurement series, the class inputs can be left empty.
        ========
        INPUT:\n
        ... Can be left empty when dealing with the static measurement series \n
        ... fileName [String]:          This is the name of the CSV file containing all dynamic measurements, choose between reference and actual \n
        ... t0 [Value]:                 At what time do you want the stationary flight condition variables? \n
        ... SI [Boolean]:               Set to True for SI units, False for non-SI units \n

        OUTPUT:\n
        ... ParametersAsymmetric [Class]:      Class containing the parameters\n
        '''

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
        aeroCoeff = calc1.calcAeroCoeff(fileName)
        self.e = aeroCoeff[2] # Oswald factor [ ]
        self.CD0 = aeroCoeff[3] # Zero lift drag coefficient [ ]
        self.CLa = aeroCoeff[0] # Slope of CL-alpha curve [ ]

        # Longitudinal stability
        self.Cmde = calc2.calcElevEffectiveness(fileName)
        self.Cma     = calc2.calcElevDeflection(fileName)[1]

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
        self.Clda = -0.28
        self.Cldr = +0.03

        self.Cnb = +0.106
        self.Cnbdot = 0
        self.Cnp = -0.08
        self.Cnr = -0.195
        self.Cnda = -0.035
        self.Cndr = -0.085


class ParametersSymmetric:
    '''
        DESCRIPTION:    Class containing all symmetric matched parameters. To find the constant parameters at a certain time during the dynamic measurements, give inputs to this class. For the static measurement series, the class inputs can be left empty.
        ========
        INPUT:\n
        ... Can be left empty when dealing with the static measurement series \n
        ... fileName [String]:          This is the name of the CSV file containing all dynamic measurements, choose between reference and actual \n
        ... t0 [Value]:                 At what time do you want the stationary flight condition variables? \n
        ... SI [Boolean]:               Set to True for SI units, False for non-SI units \n

        OUTPUT:\n
        ... ParametersSymmetric [Class]:      Class containing the parameters\n
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
        aeroCoeff = calc1.calcAeroCoeff(fileName)
        self.e = aeroCoeff[2] # Oswald factor [ ]
        self.CD0 = aeroCoeff[3] # Zero lift drag coefficient [ ]
        self.CLa = aeroCoeff[0] # Slope of CL-alpha curve [ ]

        # Longitudinal stability
        self.Cmde = calc2.calcElevEffectiveness(fileName)
        self.Cma = calc2.calcElevDeflection(fileName)[1]

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
        self.CZu = -0.376160
        self.CZa = -5.74340
        self.CZadot = -0.00350
        self.CZq = -5.66290
        self.CZde = -1.7403

        self.Cmu = +0.12582
        self.Cmadot = +0.17800
        self.Cmq = -12.75151
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
    inputFile = input("Do you want reference or actual data?")
    while inputFile not in ['reference', 'actual']:
        inputFile = input("Invalid input: choose between 'reference' or 'actual'")
    showPlot = input("Do you want nice plots of flight test data?")
    while showPlot not in ['Yes', 'No','yes','no','y','n']:
         showPlot = input("Invalid input: choose between 'Yes' or 'No'")
    doSimulate = input("Do you want nice plots of simulation?")
    while doSimulate not in ['Yes', 'No','yes','no','y','n']:
        doSimulate = input("Invalid input: choose between 'Yes' or 'No'")
    doImprove = input("Do you want the improved parameters?")
    while doImprove not in ['Yes', 'No', 'yes', 'no','y','n']:
        doImprove = input("Invalid input: choose between 'Yes' or 'No'")
    doEigen = input("Do you want the numerical, analytical and flight test data eigenvalues?")
    while doEigen not in ['Yes', 'No', 'yes', 'no','y','n']:
        doEigen = input("Invalid input: choose between 'Yes' or 'No'")

    if doSimulate in ['Yes','yes','y']:
        plotNumerical = True
    else:
        plotNumerical = False

    #-----------------------------------------------------
    # Create time and duration classes
    #-----------------------------------------------------
    tStart = StartTime(fileName=inputFile)
    tDuration = DurationTime(fileName=inputFile)

    #-----------------------------------------------------
    # Create parameters classes
    #-----------------------------------------------------
    if doImprove in ['Yes','yes','y']:
        paramPhugoid = ParametersSymmetric(fileName=inputFile,t0=tStart.tPhugoid,SI=True)
        paramShortPeriod = ParametersSymmetric(fileName=inputFile,t0=tStart.tShortPeriod,SI=True)
        paramDutchRoll = ParametersAsymmetric(fileName=inputFile,t0=tStart.tDutchRoll,SI=True)
        paramDutchRollYD = ParametersAsymmetric(fileName=inputFile, t0=tStart.tDutchRollYD,SI=True)
        paramAperRoll = ParametersAsymmetric(fileName=inputFile, t0=tStart.tAperRoll,SI=True)
        paramSpiral = ParametersAsymmetric(fileName=inputFile, t0=tStart.tSpiral,SI=True)
    elif doImprove in ['No','no','n']:
        paramPhugoid = ParametersOld(fileName=inputFile,t0=tStart.tPhugoid,SI=True)
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

    #-----------------------------------------------------
    # Calculate numerical, analytical and flight test data eigenvalues
    #-----------------------------------------------------
    if doEigen in ['Yes','yes','y']:
        eigPhugoid,eigPhugoidSimp = calcEigenPhugoid(paramPhugoid)
        eigShortPeriod = calcEigenShortPeriod(paramShortPeriod)
        eigDutchRoll,eigDutchRollSimp = calcEigenDutchRoll(paramDutchRoll)
        eigDutchRollYD, eigDutchRollYDSimp = calcEigenDutchRoll(paramDutchRoll)
        eigAperRoll = calcEigenAperRoll(paramAperRoll)
        eigSpiral = calcEigenSpiral(paramSpiral)

        print("Eigenvalues numerical Phugoid: ",ssPhugoid.Eigs[2].eig,ssPhugoid.Eigs[3].eig)
        print("Periods numerical Phugoid",ssPhugoid.Eigs[2].P,ssPhugoid.Eigs[3].P)
        print("Thalf numerical Phugoid",ssPhugoid.Eigs[2].Thalf,ssPhugoid.Eigs[3].Thalf,"\n")
        print("Eigenvalues analytical Phugoid:",eigPhugoid[0].eig,eigPhugoid[1].eig)
        print("Periods analytical Phugoid",eigPhugoid[0].P,eigPhugoid[1].P)
        print("Thalf analytical Phugoid",eigPhugoid[0].Thalf,eigPhugoid[1].Thalf,"\n")

        print("Eigenvalues numerical Short Period: ", ssShortPeriod.Eigs[0].eig,ssShortPeriod.Eigs[1].eig)
        print("Periods numerical Short Period",ssShortPeriod.Eigs[0].P,ssShortPeriod.Eigs[1].P)
        print("Thalf numerical Short Period",ssShortPeriod.Eigs[0].Thalf,ssShortPeriod.Eigs[1].Thalf,"\n")
        print("Eigenvalues analytical Short Period:", eigShortPeriod[0].eig,eigShortPeriod[1].eig)
        print("Periods analytical Short Period",eigShortPeriod[0].P,eigShortPeriod[1].P)
        print("Thalf analytical Short Period",eigShortPeriod[0].Thalf,eigShortPeriod[1].Thalf,"\n")

        print("Eigenvalues numerical Dutch Roll:", ssDutchRoll.Eiga[1].eig,ssDutchRoll.Eiga[2].eig)
        print("Periods numerical Dutch Roll",ssDutchRoll.Eiga[1].P,ssDutchRoll.Eiga[2].P)
        print("Thalf numerical Dutch Roll",ssDutchRoll.Eiga[1].Thalf,ssDutchRoll.Eiga[2].Thalf,"\n")
        print("Eigenvalues analytical Dutch Roll:", eigDutchRoll[0].eig,eigDutchRoll[1].eig)
        print("Periods analytical Dutch Roll",eigDutchRoll[0].P,eigDutchRoll[1].P)
        print("Thalf analytical Dutch Roll",eigDutchRoll[0].Thalf,eigDutchRoll[1].Thalf,"\n")

        print("Eigenvalues numerical Dutch Roll YD:", ssDutchRollYD.Eiga[1].eig,ssDutchRollYD.Eiga[2].eig)
        print("Eigenvalues analytical Dutch Roll YD:", eigDutchRollYD[0].eig,eigDutchRollYD[1].eig,"\n")

        print("Eigenvalues numerical Aperiodic Roll",ssAperRoll.Eiga[0].eig)
        print("Periods numerical Aperiodic Roll",ssAperRoll.Eiga[0].P)
        print("Thalf numerical Aperiodic Roll",ssAperRoll.Eiga[0].Thalf,"\n")
        print("Eigenvalues analytical Aperiodic Roll:",eigAperRoll.eig)
        print("Periods analytical Aperiodic Roll",eigAperRoll.P)
        print("Thalf analytical Aperiodic Roll",eigAperRoll.Thalf,"\n")

        print("Eigenvalues numerical Spiral:", ssSpiral.Eiga[3].eig)
        print("periods numerical Spiral",ssSpiral.Eiga[3].P)
        print("Thalf numerical Spiral",ssSpiral.Eiga[3].Thalf,"\n")
        print("Eigenvalues analytical Spiral:" ,eigSpiral.eig)
        print("Periods analytical Spiral",eigSpiral.P)
        print("Thalf analytical Spiral",eigSpiral.Thalf,"\n")

        re, im = eigen_dyn_phugoid('ubar', paramPhugoid)
        re1, im1 = eigen_dyn_phugoid('q', paramPhugoid)
        re2, im2 = eigen_dyn_phugoid('theta_stab', paramPhugoid)
        re3, im3 = eigen_dyn_dutchroll('p', paramDutchRoll)
        re4, im4 = eigen_dyn_dutchroll('r', paramDutchRoll)

        print("Eigenvalue flight test Phugoid for ubar:", re, 'i', im)
        print("Eigenvalue flight test Phugoid for q:", re1, 'i', im1)
        print("Eigenvalue flight test Phugoid for theta_stab:", re2, 'i', im2)
        print("Eigenvalue flight test Dutchroll for p:", re3, 'i', im3)
        print("Eigenvalue flight test Dutchroll for r:", re4, 'i', im4)

    #-----------------------------------------------------
    # Plot eigen motions from flight test data or reference data
    #-----------------------------------------------------
    if showPlot in ['Yes','yes','y']:
        plotMotionsTest(paramPhugoid,inputFile,tStart.tPhugoid,tDuration.tPhugoid,ssPhugoid,'phugoid',plotNumerical,SI=True)  # plot from reference data for phugoid
        plotMotionsTest(paramShortPeriod, inputFile, tStart.tShortPeriod, tDuration.tShortPeriod, ssShortPeriod, 'short period', plotNumerical, SI=True)  # plot from reference data for short period
        plotMotionsTest(paramDutchRoll, inputFile, tStart.tDutchRoll, tDuration.tDutchRoll, ssDutchRoll, 'dutch roll',plotNumerical, SI=True)
        plotMotionsTest(paramDutchRollYD,inputFile,tStart.tDutchRollYD,tDuration.tDutchRollYD,ssDutchRollYD,'dutch roll yd',plotNumerical,SI=True)
        plotMotionsTest(paramAperRoll, inputFile, tStart.tAperRoll, tDuration.tAperRoll, ssAperRoll, 'aper roll', plotNumerical,SI=True)
        plotMotionsTest(paramSpiral, inputFile, tStart.tSpiral, tDuration.tSpiral, ssSpiral, 'spiral', plotNumerical,SI=True)
        plt.show()




    # for verification

    # plot_initial_value_Response(ssPhugoid, 'symmetric')
    # plot_initial_value_Response(ssDutchRoll, 'asymmetric')
    # plotVerificationStepResponse(ssPhugoid,'symmetric')
    # plotVerificationStepResponse(ssDutchRoll,'asymmetric')
    # plotVerificationImpulseResponse(ssPhugoid,'symmetric')
    # plotVerificationImpulseResponse(ssDutchRoll,'asymmetric')

if __name__ == "__main__":
    #this is run when script is started, dont change
    main()

