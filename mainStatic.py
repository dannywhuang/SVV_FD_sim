import staticCalc1 as calc1
import staticCalc2 as calc2


def main():
    inputFile = input("Choose to evaluate the 'reference' or 'actual' data: ")
    while inputFile not in ['reference', 'actual']:
        inputFile = input("Invalid input: choose between 'reference' or 'actual'")

    # =============================
    # Aircraft Parameters
    # =============================
    # Aerodynamic parameters
    # ....

    # Stability parameters
    Cmdelta = calc2.calcElevEffectiveness(inputFile)
    Cma     = calc2.calcElevDeflection(inputFile)[1]

    print("\nThe elevator effectiveness Cmdelta for the "+inputFile+" data equals: ", round(Cmdelta,3))
    print("\nThe longitudinal stability Cma for the "+inputFile+" data equals: ", round(Cma,3))

    # =============================
    # Aircraft Plots
    # =============================
    # Aerodynamic plots
    # ....
    
    # Stability plots
    # ....


if __name__ == "__main__":
    #this is run when script is started, dont change
    main()
