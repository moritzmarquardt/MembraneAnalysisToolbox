import matplotlib.pyplot as plt
import numpy as np

from MembraneAnalysisToolbox.MembraneStructures import (
    HexagonalMembrane,
    Membrane,
    MembraneForDiffusionAnalysis,
)
from MembraneAnalysisToolbox.PoreAnalysis import PoreAnalysis

membrane = HexagonalMembrane(L=180, selectors="resname C")


PA = PoreAnalysis(
    topology_file="/bigpool/users/st166545/MembraneAnalysisToolbox/tests/data/hex_box_hex_dod/topol.tpr",
    trajectory_file="/bigpool/users/st166545/MembraneAnalysisToolbox/tests/data/hex_box_hex_dod/traj.xtc",
    membrane=membrane,
    verbose=True,
)

PA.find_membrane_location()
PA.print_membrane_location()
PA.verify_membrane_location()

print(PA.membrane.lowerZ)

z_vals_c = PA.trajectories["resname C"][:, :, 2].flatten()
hist, hist_edges = np.histogram(z_vals_c, bins=100, density=True)
x = hist_edges[:-1] + np.diff(hist_edges) / 2
y = hist
plt.plot(x, y)
plt.axvline(PA.membrane.lowerZ, color="red")
plt.show()
