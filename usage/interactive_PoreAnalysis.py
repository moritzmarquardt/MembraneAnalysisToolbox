import matplotlib.pyplot as plt

from MembraneAnalysisToolbox.MembraneStructures import (
    CubicMembrane,
    HexagonalMembrane,
    Solvent,
)
from MembraneAnalysisToolbox.PoreAnalysis import PoreAnalysis

print("\n\nInteractive Analysis of the Membrane Pore")
print("===============================================")
print("\nFirst enter the paths to the simulation files.")
print("Remember to end the path with a '/'.")
print("\nEnter the path to the topol.tpr file: ")
topol_path = input("-> ")
print("\nEnter Filename (press ENTER if it is topol.tpr): ")
topol_file_name = input("-> ")
if topol_file_name == "":
    topol_file_name = "topol.tpr"
topol_file = topol_path + topol_file_name
print(
    "\nEnter the path to the traj.xtc file (press ENTER if it is in the same path as topol.tpr): "
)
traj_path = input("-> ")
if traj_path == "":
    traj_path = topol_path
print("\nEnter Filename (press ENTER if it is traj.xtc): ")
traj_file_name = input("-> ")
if traj_file_name == "":
    traj_file_name = "traj.xtc"
traj_file = traj_path + traj_file_name

print("\nEntere the type of membrane in the Simulation. Possible types are:")
print("'C': Cubic Membrane \n'H': Hexagonal Membrane \n'S': Solvent")
membrane_type = input("-> ")

match membrane_type:
    case "C":
        structure = CubicMembrane(
            selectors="resname C",
            cube_arrangement=(2, 2, 2),
            cube_size=90,
            pore_radius=15,
        )
    case "H":
        print(
            "\nEnter the length of the membrane in Angstrom or press ENTER to select the default of 180: "
        )
        L = input("-> ")
        L = 180 if L == "" else int(L)
        structure = HexagonalMembrane(
            selectors="resname C",
            L=L,
        )
    case "S":
        print("\nEnter the lower Z value: ")
        lowerZ = int(input("-> "))
        print("\nEnter the upper Z value: ")
        upperZ = int(input("-> "))
        L = upperZ - lowerZ
        structure = Solvent(
            lowerZ=lowerZ,
            upperZ=upperZ,
            L=L,
        )
    case _:
        raise ValueError("Invalid input for membrane_type")

# Check if the user wants to save the results and if so, where
results_dir = None
print("\nDo you want to save the results? ('y' or press ENTER for no)")
want_to_save_results = input("-> ") == "y"

if want_to_save_results:
    print(
        "\nSave results path (has to end with '/'; press ENTER to skip and save to standard /analyis folder in Simulation folder): "
    )
    results_dir_input = input("-> ")
    if results_dir_input == "":
        results_dir = traj_path + "analysis/"
    else:
        results_dir = results_dir_input


PA = PoreAnalysis(
    topology_file=topol_file,
    trajectory_file=traj_file,
    membrane=structure,
    analysis_max_step_size_ps=1000,
    results_dir=results_dir,
    verbose=True,
)

print(PA)
if isinstance(PA.membrane, Solvent):
    PA.print_membrane_location()
else:
    PA.find_membrane_location()
    PA.print_membrane_location()
    PA.verify_membrane_location()
plt.show()

wants_to_analyse = True
while wants_to_analyse:
    print("\nWhat do you want to analyse?")
    print("\tA: Effective Pore Radius")
    print("\tB: Density (KDE) of the Solvent & Membrane normed")
    print("\tC: Density (KDE) of the Solvent")
    print("\tQ: Quit")

    analysis_type = input("-> ")
    match analysis_type:
        case "A":
            PA.analyseConstraints(
                "resname C",
                y_constraints=(25, 35),
                z_constraints=PA.membrane.find_zConstraints(),
            )
            plt.show()
            print(
                "\nWhat y_constraints do you want to use for the analysis? Example '25, 35'. This means that the analysis will be done between 25 and 35 Angstrom in the y direction."
            )
            y_constraints = input("-> ")
            y_constraints = y_constraints.split(", ")
            edges = PA.calculateEffectivePoreSize(
                solvent_selectors=["resname HEX and name C1"],
                z_constraints=PA.membrane.find_zConstraints(),
                y_constraints=(25, 35),
                strategy="intersection",
                bins=50,
            )
            plt.show()
            print(edges)
            print(f"The effective pore size is: {edges[1] - edges[0]}")
        case "B":
            print(
                "\nWhich atoms would you like to consider for the normed density plot? Example: 'resname DOD and name C3' or 'resname C, resname DOD and name C2'"
            )
            atoms = input("-> ")
            selectors = atoms.split(", ")
            PA.analyseDensityNormed(
                selectors=selectors,
                z_range=PA.membrane.find_zConstraints(),
                skip=10,
                bw=0.13,
            )
            plt.show()
        case "C":
            print(
                "\nWhich atoms would you like to consider for the density plot? Example: 'resname DOD and name C3' or 'resname HEX and name C1, resname DOD and name C2'"
            )
            atoms = input("-> ")
            selectors = atoms.split(", ")
            PA.analyseDensity(
                selectors=selectors,
                z_range=PA.membrane.find_zConstraints(),
                skip=10,
                bw=0.13,
            )
            plt.show()
        case "Q":
            break
        case _:
            print("Invalid input, try again.")
            wants_to_analyse = (
                input(
                    "\nDo you want to analyse something else? ('y' or press ENTER for no)"
                )
                == "y"
            )

    wants_to_analyse = (
        input("\nDo you want to analyse something else? ('y' or press ENTER for no)")
        == "y"
    )
