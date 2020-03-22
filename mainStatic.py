import matplotlib.pyplot as plt

import staticCalc1 as calc1
import staticCalc2 as calc2



def main():

    inputFile = input("\nChoose to evaluate the 'reference' or 'actual' data: ")
    while inputFile not in ['reference', 'actual']:
        inputFile = input("Invalid input: choose between 'reference' or 'actual'")

    # =============================
    # Aircraft Parameters
    # =============================
    # Aerodynamic parameters
    print("\n ============ Aerodynamic Parameters ============")
    CLa  = calc1.calcAeroCoeff(inputFile,'static1')[0]
    aoa0 = calc1.calcAeroCoeff(inputFile,'static1')[1]
    e    = calc1.calcAeroCoeff(inputFile,'static1')[2]
    CD0  = calc1.calcAeroCoeff(inputFile,'static1')[3]

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
    # Aerodynamic plots
    print("\n ============ Aerodynamic Plots ============")
    calc1.plotPolar(inputFile)

    # Stability plots
    print("\n ============ Stability Plots ============")
    plt.figure('Elevator Trim Curve',[10,7])
    calc2.plotElevTrimCurve(inputFile)

    plt.figure('Elevator Force Control Curve',[10,7])
    calc2.plotElevContrForceCurve(inputFile)

    plt.show()

    # =============================
    # Code to run mainStatic again
    # =============================
    question = input("\nDo you want to run the program again? [y/n]: ")
    while question not in ['y','n']:
        question = input("\nInvalid response, choose between 'y' or 'n'")
    if question == 'y':
        main()
    if question == 'n':
        print('\nTot de volgende keer!!!\n')
    return


if __name__ == "__main__":
    # This is run when script is started, dont change
    main()
