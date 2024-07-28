import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from statsmodels.distributions.empirical_distribution import ECDF

from MembraneAnalysisToolbox.core_functions import (
    calculate_diffusion,
    dur_dist_improved,
    findPassages,
    fitfunc_hom,
    fitfunc_hom_cdf,
    save_1darr_to_txt,
)
from MembraneAnalysisToolbox.MembraneAnalysis import MembraneAnalysis
from MembraneAnalysisToolbox.MembraneStructures import Membrane


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

    length is always in Angstrom
    time is always in ps (exept in the diff calculation) This choice is done because MDA works with ps
    """

    def __init__(
        self,
        topology_file: str,
        trajectory_file: str,
        membrane: Membrane,
        analysis_max_step_size_ps: int = None,
        results_dir: str = None,
        verbose: bool = True,
    ):
        super().__init__(
            topology_file=topology_file,
            trajectory_file=trajectory_file,
            analysis_max_step_size_ps=analysis_max_step_size_ps,
            results_dir=results_dir,
            verbose=verbose,
            membrane=membrane,
        )
        self.passageTimes = {}
        self.passageStarts = {}
        self.passageIndices = {}
        self.n_passages = {}
        self.D = {}

        u_sim_time = self.u.trajectory.n_frames * self.u.trajectory.dt
        ana_sim_time = self.n_frames * self.step_size
        if abs(u_sim_time - ana_sim_time) > self.analysis_max_step_size_ps:
            Exception(
                f"Simulation time of trajectory ({u_sim_time}) and analysed time ({ana_sim_time}) do not match."
            )

    def __str__(self):
        return (
            f"DiffusionAnalysis object:\n"
            f"  topology_file: {self.topology_file}\n"
            f"  trajectory_file: {self.trajectory_file}\n"
            f"  results_dir: {self.results_dir}\n"
            f"  Membrane: {self.membrane}\n"
            f"  verbose: {self.verbose}\n"
            f"  trajectories: {self.trajectories.keys()}\n"
            f"  results: \n"
            f"      D: {self.D}\n"
            f"      passageTimes: {self.passageTimes}\n"
            f"      passageStarts: {self.passageStarts}\n"
            f"      passageIndices: {self.passageIndices}\n"
            f"      n_passages: {self.n_passages}\n"
            f"  Simulation footprints: \n"
            f"      u_trajectory_dt (ps): {self.u.trajectory.dt}\n"
            f"      u_sim_time (ps): {(self.u.trajectory.n_frames - 1) * self.u.trajectory.dt}\n"
            f"      analysis_max_step_size_ps (ps): {self.analysis_max_step_size_ps}\n"
            f"      actual analysed step_size (ps): {self.step_size}\n"
            f"      nth_frame: {self.nth_frame}\n"
            f"      n_frames analysed: {self.n_frames}\n"
            f"      ana_sim_time (ps): {(self.n_frames - 1) * self.step_size}\n"
            f"      u: {self.u}\n"
        )

    def find_membrane_location(self):
        self._allocateTrajectories(self.membrane.selector)
        self.membrane.find_location(self.trajectories[self.membrane.selector])

    def print_membrane_location(self):
        self.membrane.print_location()

    def verify_membrane_location(self):
        self._allocateTrajectories(self.membrane.selector)
        self.membrane.plot_location(self.trajectories[self.membrane.selector])

    def calc_passagetimes(self, selector: str):
        """
        Calculates the passage times for the given selectors.
        store it in ps in self.passageTimes[selector]

        Args:
            selectors (str): The selectors to calculate passage times for.

        """
        self._allocateTrajectories(selector)

        isAtomAbove, isAtomBelow = self.membrane.define_isabove_isbelow_funcs()

        T = self.trajectories[selector]
        ffs, ffe, ffi = findPassages(T, isAtomAbove, isAtomBelow)

        # convert timesteps to ps
        ffs_ps = ffs * self.u.trajectory.dt * self.nth_frame
        ffe_ps = ffe * self.u.trajectory.dt * self.nth_frame

        # store the results
        self.passageTimes[selector] = ffe_ps - ffs_ps
        self.passageStarts[selector] = ffs_ps
        self.passageIndices[selector] = ffi
        self.n_passages[selector] = len(ffs_ps)

    def plot_passagetimedist(self, selector: str):
        if selector not in self.passageTimes.keys():
            raise ValueError(
                "Passage times for the selector must be calculated before plotting the distribution."
            )
        passage_times = self.passageTimes[selector] / 1000  # convert to ns
        fig, ax = super()._create_histogram(
            passage_times,
            label=None,
            bins=100,
            title="Verteilung der Durchgangszeiten",
            xlabel="Durchgangszeiten in ns",
            ylabel="relative Häufigkeit",
        )
        return fig, ax

    def calc_diffusion(self, selector: str):
        if self.membrane.L is None:
            raise ValueError(
                "L must be set before calculating the diffusion coefficient."
            )

        if selector not in self.passageTimes.keys():
            raise ValueError(
                "Passage times for the selector must be calculated before calculating the diffusion coefficient."
            )

        passage_times = self.passageTimes[selector] / 1000  # convert to ns

        self.D[selector] = calculate_diffusion(self.membrane.L, passage_times)

    def plot_diffusion(self, selector: str):
        if self.membrane.L is None:
            raise ValueError(
                "L must be set before calculating the diffusion coefficient."
            )
        if selector not in self.passageTimes.keys():
            raise ValueError(
                "Passage times for the selector must be calculated before plotting the diffusion coefficient."
            )
        if selector not in self.D.keys():
            raise ValueError(
                "Diffusion coefficient for the selector must be calculated before plotting the diffusion coefficient."
            )

        passage_times = self.passageTimes[selector] / 1000  # convert to ns
        D = self.D[selector]

        ecdf = ECDF(passage_times)
        idx = (np.abs(ecdf.y - 0.5)).argmin()
        centertime = ecdf.x[idx]

        """ PLOT DATA """
        x_lim = centertime * 4
        x_cdf = np.linspace(0, x_lim * 2, 200)
        y_hom_cdf = fitfunc_hom_cdf(x_cdf, D, self.membrane.L)

        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle("PDF and CDF fit")
        ax2.scatter(ecdf.x, ecdf.y, color=[0, 0.5, 0.5], label="CDF data points")
        ax2.plot(
            x_cdf[1:],
            y_hom_cdf[1:],
            label="hom",
            color="red",
            ls="dashed",
        )
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
        y2 = fitfunc_hom(x, D, self.membrane.L)

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
            ffs, ffe, _ = dur_dist_improved(
                piece, [self.z_lower, self.z_lower + self.membrane.L]
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
                ffs, ffe, _ = dur_dist_improved(sample, [z_lower, z_lower + L])
                bootstrap_diffusions[i] = self.calc_diffusion(ffe - ffs, L, plot=plot)
            except Exception as e:
                print(e)
                bootstrap_diffusions[i] = np.nan

        if self.verbose:
            print("\nBootstrapping finished.")

        return bootstrap_diffusions

    def store_results_json(self):
        out = {
            "L": self.membrane.L,
            "z_lower": self.z_lower,
            "D": self.D,
            "n_passages": self.n_passages,
        }
        # print(out)
        self._store_dict_as_json(
            out, self.results_dir + "diffusion_analysis_results.json"
        )

    def save_passage_times_in_ns_to_txt(self, selector: str, file_name: str):
        if not file_name.endswith(".txt"):
            raise ValueError("File name must end with .txt.")
        save_1darr_to_txt(
            self.passageTimes[selector] / 1000,  # convert to ns
            self.results_dir + file_name,
        )

    @staticmethod
    def _store_dict_as_json(dictionary, filename):
        with open(filename, "w") as f:
            json.dump(dictionary, f)

    # TODO implement this function correctly
    def create_rand_passages_plot(self, selector: str, n: int) -> tuple:
        self._allocateTrajectories(selector)

        fig = plt.figure("3d trajektorien")
        ax = fig.add_subplot(projection="3d")

        print(self.passageIndices[selector].shape[0])
        rand_indices = np.random.randint(
            0, self.passageIndices[selector].shape[0], size=(n)
        )
        print(rand_indices)
        # transform ns back to frames by dividing by the step size
        ffss = self.passageStarts[selector][rand_indices] / self.step_size
        print(ffss)
        ffss = ffss.astype(int)
        print(ffss)
        ffes = ffss + (self.passageTimes[selector][rand_indices] / self.step_size)
        print(ffes)
        ffes = ffes.astype(int)
        print(ffes)
        ffis = self.passageIndices[selector][rand_indices]
        print(ffis)

        x_passages = self.trajectories[selector][ffis, :, 0]
        y_passages = self.trajectories[selector][ffis, :, 1]
        z_passages = self.trajectories[selector][ffis, :, 2]

        for sel in range(n):
            slicer = slice(ffss[sel] - 1, ffes[sel] + 2, 1)
            x_passages_sel = x_passages[sel, slicer]
            y_passages_sel = y_passages[sel, slicer]
            z_passages_sel = z_passages[sel, slicer]
            ax.plot(
                x_passages_sel,
                y_passages_sel,
                z_passages_sel,
            )  # ++1 so that the last timestep is also included
        ax.scatter(
            x_passages[:, 0],
            y_passages[:, 0],
            z_passages[:, 0],
        )  # starting point
        ax.scatter(
            x_passages[:, -1],
            y_passages[:, -1],
            z_passages[:, -1],
        )  # end point
        return fig, ax

    # TODO: implement this function correctly
    def plot_starting_points(self, selector: str):
        res = selector.split(" ")[1]
        fig = plt.figure("plotten aller Startpunkte")
        ax = fig.add_subplot(projection="3d")
        ax.set_xlabel("x in nm", fontsize="x-large")
        ax.set_ylabel("y in nm", fontsize="x-large")
        ax.set_zlabel("z in nm", fontsize="x-large")
        ax.set_title(
            "Membrane-entry points of the passage-trajectories (" + res.upper() + ")",
            fontsize="x-large",
        )
        x_passages = self.trajectories[selector][self.passageIndices[selector], :, 0]
        y_passages = self.trajectories[selector][self.passageIndices[selector], :, 1]
        z_passages = self.trajectories[selector][self.passageIndices[selector], :, 2]
        ffs = self.passageStarts[selector] / self.step_size
        ffs = ffs.astype(int)
        ax.scatter(
            x_passages[np.arange(np.size(x_passages, 0)), ffs + 1] / 10,
            y_passages[np.arange(np.size(x_passages, 0)), ffs + 1] / 10,
            z_passages[np.arange(np.size(x_passages, 0)), ffs + 1] / 10,
        )  # ugly way of getting the point. maybe there is a better way"""
        return fig, ax
