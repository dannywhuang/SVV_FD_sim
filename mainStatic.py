import matplotlib.pyplot as plt

import staticCalc1 as calc1
import staticCalc2 as calc2


def main():
    inputFile = input("\nChoose to evaluate the 'reference' or 'actual' data: ")
    while inputFile not in ['reference', 'actual']:
        inputFile = input("\nInvalid input: choose between 'reference' or 'actual'")

    plots = input("\nDo you want to see plots [y/n]: ")
    while plots not in ['y','n']:
        plots = input("\nInvalid input: choose between 'y' or 'n'")

    # =============================
    # Aircraft Parameters
    # =============================
    # Aerodynamic parameters
    print("\n ============ Aerodynamic Parameters ============")
    CLa  = calc1.calcAeroCoeff(inputFile)[0]
    aoa0 = calc1.calcAeroCoeff(inputFile)[1]
    e    = calc1.calcAeroCoeff(inputFile)[2]
    CD0  = calc1.calcAeroCoeff(inputFile)[3]

    print("\nThe lift curve slope for the "+inputFile+" data equals: CLa = ",round(CLa,3))
    print("\nThe zero-lift angle of attack for the " + inputFile + " data equals: aoa0 = ", round(aoa0,3))
    print("\nThe Oswald efficiency factor for the " + inputFile + " data equals: e = ", round(e,3))
    print("\nThe zero-lift drag coefficient for the " + inputFile + " data equals: CD0 = ", round(CD0,3))

    # Stability parameters
    print("\n ============ Stability Parameters ============")
    Cmdelta = calc2.calcElevEffectiveness(inputFile)
    Cma     = calc2.calcElevDeflection(inputFile)[1]

    print("\nThe elevator effectiveness for the "+inputFile+" data equals: Cmdelta =", round(Cmdelta,3))
    print("\nThe longitudinal stability for the "+inputFile+" data equals: Cma =", round(Cma,3))

    # =============================
    # Aircraft Plots
    # =============================
    if plots == 'y':
        # Aerodynamic plots
        calc1.plotLift(inputFile)
        calc1.plotPolar(inputFile)

        # Stability plots
        calc2.plotRedElevTrimCurve(inputFile)
        calc2.plotRedElevContrForceCurve(inputFile)

        print("\nTot de volgende keer!!!\n")

        plt.show()
    elif plots == 'n':
        print('\nTot de volgende keer!!!\n')
    return


if __name__ == "__main__":
    # This is run when script is started, dont change
    main()
