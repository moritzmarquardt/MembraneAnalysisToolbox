import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from statsmodels.distributions.empirical_distribution import ECDF

import MembraneAnalysisToolbox.funcs as tfm
import MembraneAnalysisToolbox.plot as tfmp
from MembraneAnalysisToolbox.MembraneAnalysis import MembraneAnalysis


class DiffusionAnalysis(MembraneAnalysis):
    """
    This class represents an analysis on the trajectory data of a molecular
    dynamics simulation. It is designed to analyse systems with molecules/beads
    that represent a membrane and molecules/beads that represent a solvent
    that is surrounding and penetrating the membrane.
    The analysis class offers methods to determine properties of that membrane
    such as
        - the passage time ditribution of the solvent molecules through the
        membrane
        - the diffusion coefficient of the solvent molecules in the membrane
    These properties are stored in the class object and can be saved in a
    results directory.
    """

    def __init__(
        self,
        topology_file: str,
        trajectory_file: str,
        analysed_max_step_size_ps: int = None,
        results_dir: str = None,
        verbose: bool = True,
        L: float = None,
        z_lower: float = None,
        D: float = None,
    ):
        super().__init__(
            topology_file=topology_file,
            trajectory_file=trajectory_file,
            analysed_max_step_size_ps=analysed_max_step_size_ps,
            results_dir=results_dir,
            verbose=verbose,
        )
        self.L = L
        self.z_lower = z_lower
        self.passageTimes = {}
        self.D = {}

    def __str__(self):
        return (
            f"DiffusionAnalysis object:\n"
            f"  L: {self.L}\n"
            f"  z_lower: {self.z_lower}\n"
            f"  D: {self.D}\n"
            f"  results_dir: {self.results_dir}\n"
            f"  topology_file: {self.topology_file}\n"
            f"  trajectory_file: {self.trajectory_file}\n"
            f"  verbose: {self.verbose}\n"
            f"  analysed_max_step_size_ps: {self.analysed_max_step_size_ps}\n"
            f"  nth_frame: {self.nth_frame}\n"
            f"  n_frames: {self.n_frames}\n"
            f"  u: {self.u}\n"
            f"  trajectories: {self.trajectories.keys()}\n"
        )

    def find_membrane_location_hexstructure(self, mem_selector: str):
        z_lower = super().find_membrane_location_hexstructure(mem_selector, self.L)
        self.z_lower = z_lower

    def calc_passagetimes(self, selector: str):
        """
        Calculates the passage times for the given selectors.

        Args:
            selectors (str): The selectors to calculate passage times for.
            z_lower (float): The lower bound of the z-coordinate range.
            L (float): The length of the z-coordinate range.

        Returns:
            tuple: A tuple containing three arrays - concatenated first-first passage times (ffs),
                   concatenated first-final passage times (ffe), and concatenated indices (indizes).
        """
        self._allocateTrajectories(selector)

        if self.z_lower is None:
            raise ValueError(
                "z_lower must be set before calculating passage times."
                "This can be done by calling find_membrane_location_hexstructure."
            )
        if self.L is None:
            raise ValueError("L must be set before calculating passage times.")

        z = self.trajectories[selector][:, :, 2]
        z_boundaries = [self.z_lower, self.z_lower + self.L]
        ffs, ffe, _ = tfm.dur_dist_improved(z, z_boundaries)

        self.passageTimes[selector] = ffe - ffs

    def plot_passagetimedist(self, selector: str):
        plt.figure("Verteilung der Durchgangszeiten")
        passage_times = self.passageTimes[selector]
        tfmp.plot_dist(passage_times, max_range=np.max(passage_times) * 1.1)
        plt.xlabel("Durchgangszeiten")
        plt.ylabel("relative Häufigkeit")

    def calc_diffusion(self, passage_times):
        """
        Calculate the diffusion coefficient using the methods of Gotthold Fläschner.

        Args:
            passage_times (array-like): Array of passage times.
            L (float): Length parameter.
            T (float): Temperature parameter.

        Returns:
            float: The diffusion coefficient.

        """
        if self.L is None:
            raise ValueError(
                "L must be set before calculating the diffusion coefficient."
            )

        ecdf = ECDF(passage_times)
        idx = (np.abs(ecdf.y - 0.5)).argmin()
        centertime = ecdf.x[idx]

        """ FIT DATA """
        params_hom_cdf = self.fitting_hom_cdf_lsq(ecdf.x[1:], ecdf.y[1:], self.L)

        D_hom_cdf = params_hom_cdf[0]

        return D_hom_cdf

    def plot_diffusion(self, passage_times, D):
        if self.L is None:
            raise ValueError(
                "L must be set before calculating the diffusion coefficient."
            )

        ecdf = ECDF(passage_times)
        idx = (np.abs(ecdf.y - 0.5)).argmin()
        centertime = ecdf.x[idx]

        """ PLOT DATA """
        x_lim = centertime * 4
        x_cdf = np.linspace(0, x_lim * 2, 200)
        y_hom_cdf = self.fitfunc_hom_cdf(x_cdf, D, self.L)

        fig, (ax1, ax2) = plt.subplots(2)
        print(fig)
        fig.suptitle("PDF and CDF fit")
        ax2.scatter(ecdf.x, ecdf.y, color=[0, 0.5, 0.5])
        ax2.plot(x_cdf[1:], y_hom_cdf[1:], label="hom", color="red", ls="dashed")
        ax2.legend(loc="center right")
        ax2.set_xlim(0, x_lim)

        # plot PDF
        """ PREPARE DATA """
        bins = int(10 * np.max(passage_times) / centertime)
        histo, edges = np.histogram(passage_times, bins, density=True)
        center = edges - (edges[2] - edges[1])
        center = np.delete(center, 0)
        edges = np.delete(edges, 0)

        """ PLOT DATA """
        x = x_cdf
        y2 = self.fitfunc_hom(x, D, self.L)

        ax1.hist(passage_times, bins=len(center), density=True, color=[0, 0.5, 0.5])
        ax1.plot(x[1:], y2[1:], label="hom", color="red", ls="dashed")
        ax1.set_xlim(0, x_lim)
        ax1.set_ylim(0, 1.2 * np.max(histo[0:]))
        ax1.legend(loc="center right")

        return fig

    def bootstrap_diffusion(self, selector, n_bootstraps, plot=True):
        # first do the bootstrapping for only one element
        if not isinstance(selector, str):
            raise ValueError("Selector must be a string.")

        self._allocateTrajectories(selector)
        bootstrap_pieces = np.array_split(
            self.trajectories[selector][:, :, 2], n_bootstraps, axis=0
        )  # idk if this works
        # print(bootstrap_pieces)
        # print(bootstrap_pieces[0].shape)
        bootstrap_diffusions = np.zeros(n_bootstraps)
        for i, piece in enumerate(bootstrap_pieces):
            ffs, ffe, _ = tfm.dur_dist_improved(
                piece, [self.z_lower, self.z_lower + self.L]
            )
            bootstrap_diffusions[i] = self.calc_diffusion(ffe - ffs)
            if plot:
                self.plot_diffusion(ffe - ffs, bootstrap_diffusions[i])
        return bootstrap_diffusions

    def bootstrapping_diffusion(
        self, selector, bootstrap_sample_length_ns, n_bootstraps, z_lower, L, plot=True
    ):
        if not isinstance(selector, str):
            raise ValueError("Selector must be a string.")

        self._allocateTrajectories(selector)

        bootstrap_sample_length = bootstrap_sample_length_ns * 1000  # convert to ps
        bootstrap_sample_length /= self.u.trajectory.dt  # convert to frames
        bootstrap_sample_length /= self.nth  # convert to steps in the analysis
        bootstrap_sample_length = int(bootstrap_sample_length)  # convert to int

        bootstrap_diffusions = np.zeros(n_bootstraps)
        rg = (
            np.random.default_rng()
        )  # this is numpys new state of the art random number generator routine according to their docs
        if self.verbose:
            print(
                f"Bootstrapping for {n_bootstraps} samples of length {bootstrap_sample_length} steps = {bootstrap_sample_length_ns}ns."
            )
        for i in range(n_bootstraps):
            sample_start_index = rg.integers(
                0, self.timesteps - bootstrap_sample_length
            )
            # print(i, sample_start_index)
            progress = int(i / n_bootstraps * 100)
            if self.verbose:
                sys.stdout.write(f"\r\tProgress: {progress}%")
                sys.stdout.flush()
            sample = self.trajectories[selector][
                :, sample_start_index : sample_start_index + bootstrap_sample_length, 2
            ]
            try:
                ffs, ffe, _ = tfm.dur_dist_improved(sample, [z_lower, z_lower + L])
                bootstrap_diffusions[i] = self.calc_diffusion(ffe - ffs, L, plot=plot)
            except Exception as e:
                print(e)
                bootstrap_diffusions[i] = np.nan

        if self.verbose:
            print("\nBootstrapping finished.")

        return bootstrap_diffusions

    #########################################################################################
    # Funktionen aus Gottholds Skript #######################################################
    def hom_cdf(self, x, D, i, L):
        t = (L) ** 2 / (i**2 * np.pi**2 * D)  # L^2/(i^2*pi^2*D)
        return (-1) ** (i - 1) * np.exp(-x / t)  # summand in Gl. 10 vanHijkoop

    def fitfunc_hom_cdf(self, x, D, L):
        i = 50  # Summe geht bis 50 (approx statt undendlich)
        result = 0
        for j in range(1, i):
            result = result + self.hom_cdf(x, D, j, L)
        return 1 - 2 * result  # gleichung 10 in vanHijkoop paper

    def fitfunc_hom_cdf_lsq(self, L):
        def f(D, x, y):
            i = 50
            result = 0
            for j in range(1, i):
                result = result + self.hom_cdf(x, D, j, L)
            return 1 - 2 * result - y

        return f

    def fitting_hom_cdf_lsq(self, x_data, y_data, L):
        res_robust = least_squares(
            self.fitfunc_hom_cdf_lsq(L),
            x0=20,
            loss="soft_l1",
            f_scale=0.3,
            args=(x_data, y_data),
        )
        return res_robust.x

    def fitfunc_hom(self, x, D, L):
        i = 151
        result = 0
        for j in range(1, i):
            result = result + self.hom(x, D, j, L)
        return 2 * np.pi**2 * D / (L) ** 2 * result

    def hom(self, x, D, i, L):
        t = (L) ** 2 / (i**2 * np.pi**2 * D)
        return (-1) ** (i - 1) * i**2 * np.exp(-x / t)


# Ende Funktionen aus Gottholds Skript ##################################################
#########################################################################################
