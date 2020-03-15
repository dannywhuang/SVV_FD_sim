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
    # ....

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
    # ....
    

    # Stability plots
    print("\n ============ Stability Plots ============")
    # ....
    return


if __name__ == "__main__":
    #this is run when script is started, dont change
    main()
