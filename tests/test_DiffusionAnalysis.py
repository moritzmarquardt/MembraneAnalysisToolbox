import unittest

from MembraneAnalysisToolbox.DiffusionAnalysis import DiffusionAnalysis
from MembraneAnalysisToolbox.MembraneStructures import (
    CubicMembrane,
    HexagonalMembrane,
    Solvent,
)

"""
Right now this is just a placeholder for the test cases that will be added later in order for the github actions to run successfully.

TODO: 
    - Add test cases for passage detection using sample membrane simulation data
    - Add test cases for Diffusion Coefficient calculation using sample membrane simulation data
"""


class TestDA(unittest.TestCase):
    def test_hex_transition_count_and_diff_coeff(self):
        topol_path = "/bigpool/users/st166545/MembraneAnalysisToolbox/tests/data/hex_box_hex_dod/"
        topol_file_name = "topol.tpr"
        topol_file = topol_path + topol_file_name
        traj_path = topol_path
        traj_file_name = "traj.xtc"
        traj_file = traj_path + traj_file_name
        L = 180
        structure = HexagonalMembrane(
            selectors="resname C",
            L=L,
        )
        results_dir = traj_path + "analysis/"
        # STEP 1: initialise the Data into the class
        DA = DiffusionAnalysis(
            topology_file=topol_file,
            trajectory_file=traj_file,
            results_dir=results_dir,
            analysis_max_step_size_ps=2,
            verbose=True,
            membrane=structure,
        )

        DA.find_membrane_location()

        DA.calc_passagetimes("resname HEX and name C1")
        DA.calc_passagetimes("resname DOD and name C2")

        # CDF test
        DA.calc_diffusion(
            "resname HEX and name C1",
            D_guess=DA.guess_D("resname HEX and name C1"),
            method="CDF",
        )
        DA.calc_diffusion(
            "resname DOD and name C2",
            D_guess=DA.guess_D("resname DOD and name C2"),
            method="CDF",
        )

        self.assertTrue(abs(DA.D["resname HEX and name C1"] - 86.29359025824535) < 1e-3)
        self.assertTrue(abs(DA.D["resname DOD and name C2"] - 35.88624329954945) < 1e-3)

        # PDF test
        DA.calc_diffusion(
            "resname HEX and name C1",
            D_guess=DA.guess_D("resname HEX and name C1"),
            method="PDF",
        )
        DA.calc_diffusion(
            "resname DOD and name C2",
            D_guess=DA.guess_D("resname DOD and name C2"),
            method="PDF",
        )

        self.assertTrue(
            abs(DA.D["resname HEX and name C1"] - 89, 96664981086349) < 1e-3
        )
        self.assertTrue(
            abs(DA.D["resname DOD and name C2"] - 37, 55927855436053) < 1e-3
        )

        self.assertEqual(DA.n_passages["resname HEX and name C1"], 2611)
        self.assertEqual(DA.n_passages["resname DOD and name C2"], 913)
        self.assertTrue(232 <= DA.membrane.lowerZ <= 234)  # exact: 233.23501586914062


if __name__ == "__main__":
    unittest.main()
