import MembraneAnalysisToolbox.TransitionPathAnalysis as TPA
import matplotlib.pyplot as plt

'''
this file is for testing and development purposes only
'''

path = "/bigpool/users/ac130484/project/finished_sim/hex/poresize/3nm_NVT/simulation_1/"

# STEP 1: initialise the Data into the class
Analysis2nm_1 = TPA.TransitionPathAnalysis( 
    topology_file = path + 'topol.tpr', 
    trajectory_file = path + 'traj.xtc',
    verbose = True
)
L = 180

# STEP 2: Find z_lower and validate by eye
z_lower = Analysis2nm_1.find_z_lower_hexstruc(mem_selector = "resname C", L=L)
print("z_lower: " + str(z_lower))
Analysis2nm_1.inspect(["resname HEX and name C1"], z_lower, L)

# STEP 3: analyse passage times and calculate diffusion coefficient
# Hex analysis
print("Hex analysis")
ffs, ffe, indizes = Analysis2nm_1.calc_passagetimes(["resname HEX and name C1"], z_lower, L)
print("passages: " + str(len(ffs)))
D = Analysis2nm_1.calc_diffusion(list(ffe-ffs), L, T = 296)
print("Diffusioncoefficient: " + str(D))
plt.show()

# Dod analysis
print("Dod analysis")
ffs, ffe, indizes = Analysis2nm_1.calc_passagetimes(["resname DOD and name C2"], z_lower, L)
print("passages: " + str(len(ffs)))
D = Analysis2nm_1.calc_diffusion(list(ffe-ffs), L, T = 296)
print("Diffusioncoefficient: " + str(D))
plt.show()
