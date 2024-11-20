import matplotlib.pyplot as plt

from MembraneAnalysisToolbox.DiffusionAnalysis import DiffusionAnalysis
from MembraneAnalysisToolbox.MembraneStructures import (
    CubicMembrane,
    HexagonalMembrane,
    Solvent,
)

print("\n\nInteractive Analysis of Diffusion in Membranes")
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

# STEP 1: initialise the Data into the class
DA = DiffusionAnalysis(
    topology_file=topol_file,
    trajectory_file=traj_file,
    results_dir=results_dir,
    analysis_max_step_size_ps=2,
    verbose=True,
    membrane=structure,
)

print(DA)

if isinstance(DA.membrane, Solvent):
    DA.print_membrane_location()
else:
    DA.find_membrane_location()
    DA.print_membrane_location()
    DA.verify_membrane_location()
plt.show()

wants_to_analyse = True
print("Analyse the transitions of atoms here:")
while wants_to_analyse:
    resname = input("Enter the resname (example: HEX): ")
    name = input("Enter the name (example: C1): ")
    selector = f"resname {resname} and name {name}"
    short = resname.lower() + "_" + name.lower()

    # perform the analysis (former method analyse_resname())
    print(f"\n{short} analysis")

    DA.calc_passagetimes(selector)
    print(f"\t{short}-passages: " + str(len(DA.passageTimes[selector])))
    # DA.plot_passagetimedist(selector)

    if want_to_save_results:
        DA.save_passage_times_in_ns_to_txt(selector, short + "_passagetimes_in_ns.txt")

    DA.calc_diffusion(selector)
    print(f"\t{short}-Diffusioncoefficient: " + str(DA.D[selector]).replace(".", ","))
    fig_diff = DA.plot_diffusion(selector)
    if want_to_save_results:
        DA.save_fig_to_results(fig=fig_diff, name="diffusion_" + short)

    DA.plot_starting_points(selector)

    plt.show()

    wants_to_analyse = (
        input(
            "\n\n\n Do you want to analyse the transitions of another resname? (y/n) "
        )
        == "y"
    )

if want_to_save_results:
    DA.store_results_json()
print(DA)
plt.show()
print("\n\n\n\n\n")
