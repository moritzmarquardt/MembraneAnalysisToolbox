import json

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

from MembraneAnalysisToolbox.core_functions import (
    findPassages,
    fit_diffusion_cdf,
    fit_diffusion_pdf,
    fitfunc_hom,
    fitfunc_hom_cdf,
    save_1darr_to_txt,
)
from MembraneAnalysisToolbox.MembraneAnalysis import MembraneAnalysis
from MembraneAnalysisToolbox.MembraneStructures import (
    CubicMembrane,
    MembraneForDiffusionAnalysis,
)


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
        membrane: MembraneForDiffusionAnalysis,
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
            f"      unique atom-names: {set(atom.name for atom in self.u.atoms)}\n"
            f"      unique resnames: {set(res.resname for res in self.u.residues)}\n"
            f"      unique combinations: {set((atom.resname, atom.name) for atom in self.u.atoms)}\n"
        )

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
        # if isinstance(self.membrane, HexagonalMembrane):
        #     ffs, ffe, ffi = findPassagesHexOptimised(
        #         T, self.membrane.lowerZ, self.membrane.upperZ
        #     )
        # else:
        #     ffs, ffe, ffi = findPassages(T, isAtomAbove, isAtomBelow)
        ffs, ffe, ffi = findPassages(T, isAtomAbove, isAtomBelow)

        # convert timesteps to ps
        ffs_ps = ffs * self.step_size
        ffe_ps = ffe * self.step_size

        # store the results
        self.passageTimes[selector] = ffe_ps - ffs_ps
        self.passageStarts[selector] = ffs_ps
        self.passageIndices[selector] = ffi
        self.n_passages[selector] = len(ffs_ps)

    def calc_passage_distances(self, selector: str):
        # function only makes sense for cubic, for o hexagonal membranes the passage lenght is trivial
        # TODO implement this for hexagonal membranes as well for consistency and so this instance test can be removed
        if not isinstance(self.membrane, CubicMembrane):
            raise ValueError(
                "This function is only implemented for cubic membranes so far."
            )
        if selector not in self.passageTimes.keys():
            raise ValueError(
                "Passage times for the selector must be calculated before calculating the passage distances."
            )
        passage_distances = np.zeros(self.n_passages[selector])
        for i, index in enumerate(self.passageIndices[selector]):
            T = self.trajectories[selector][index, :, :]
            dist = self.membrane.calc_passage_length(T)
            passage_distances[i] = dist

        return passage_distances

    def plot_passagetimedist(self, selector: str):
        """
        plots the passage time distribution for the given selector in ns
        """
        if selector not in self.passageTimes.keys():
            raise ValueError(
                "Passage times for the selector must be calculated before plotting the distribution."
            )
        passage_times = self.passageTimes[selector] / 1000  # convert to ns
        fig, ax = super()._create_histogram(
            passage_times,
            label=None,
            bins=100,
            title="Passage time distribution",
            xlabel="Passage time in ns",
            ylabel="Frequency",
        )
        return fig, ax

    def guess_D(self, selector: str):
        """
        Guess the diffusion coefficient for the given selector which can be used as an initial guess for the fit.

        Args:
            selector (str): The selector to guess the diffusion coefficient for.

        Returns:
            D_guess (float): The guessed diffusion coefficient.
        """
        if self.membrane.L is None:
            raise ValueError(
                "L must be set before calculating the diffusion coefficient."
            )

        if selector not in self.passageTimes.keys():
            raise ValueError(
                "Passage times for the selector must be calculated before calculating the diffusion coefficient."
            )

        passage_times = self.passageTimes[selector] / 1000  # convert to ns
        L = self.membrane.L

        D_guess = L**2 / (
            6 * np.mean(passage_times)
        )  # mean of eq. 9 in Hijkoop solved for D
        # this is basically a fit of means of the passage times to the mean of the FPT distribution
        # so a very rough approximation but potentially a good startiung point for a local minimisation

        return D_guess

    def calc_diffusion(self, selector: str, D_guess: float, method: str = "PDF"):
        # TODO rename to fit_diffusion_pdf()
        """
        Calculates the diffusion coefficient for the given selector.
        The diffusion coefficient is stored in self.D[selector].

        Args:
            selector (str): The selector to calculate the diffusion coefficient for.
            D_guess (float): An initial guess for the diffusion coefficient.
        """
        if self.membrane.L is None:
            raise ValueError(
                "L must be set before calculating the diffusion coefficient."
            )

        if selector not in self.passageTimes.keys():
            raise ValueError(
                "Passage times for the selector must be calculated before calculating the diffusion coefficient."
            )

        passage_times = self.passageTimes[selector] / 1000  # convert to ns

        if method == "PDF":
            self.D[selector] = fit_diffusion_pdf(
                self.membrane.L, passage_times, D_guess
            )  # in A^2/ns since L is in A and passage times are in ns
        elif method == "CDF":
            self.D[selector] = fit_diffusion_cdf(
                self.membrane.L, passage_times, D_guess
            )  # in A^2/ns since L is in A and passage times are in ns

    def plot_diffusion(self, selector: str):
        """
        Plot the Diffusion FPT fit for the given selector.

        Args:
            selector (str): The selector to plot the diffusion fit for.

        Returns:
            fig1 (matplotlib.figure.Figure): The figure with the CDF plot.
            fig2 (matplotlib.figure.Figure): The figure with the PDF plot.

        Raises:
            ValueError: If the membrane length has not been set.
            ValueError: If the passage times for the selector have not been calculated.
            ValueError: If the diffusion coefficient for the selector has not been calculated
        """
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

        # PLOT DATA
        x_lim = centertime * 4
        x_cdf = np.linspace(0, x_lim * 2, 200)
        y_hom_cdf = fitfunc_hom_cdf(x_cdf, D, self.membrane.L)

        fs = "x-large"
        figsize = (8, 3)

        fig1 = plt.figure(figsize=figsize)
        plt.xlabel("Passage time (ns)", fontsize=fs)
        plt.ylabel("cum. prob.", fontsize=fs)
        plt.scatter(ecdf.x, ecdf.y, color=[0, 0.5, 0.5], label="cum. prob.")
        plt.plot(
            x_cdf[1:],
            y_hom_cdf[1:],
            label="FPT fit (hom)",
            color="red",
            ls="dashed",
        )
        plt.legend(loc="center right", fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.xlim(0, x_lim)
        plt.tight_layout()  # so the axis label is not cut off when saving

        # plot PDF
        # PREPARE DATA
        bins = int(10 * np.max(passage_times) / centertime)
        histo, edges = np.histogram(passage_times, bins, density=True)
        center = edges - (edges[2] - edges[1])
        center = np.delete(center, 0)
        edges = np.delete(edges, 0)

        # PLOT DATA
        x = x_cdf
        y2 = fitfunc_hom(x, D, self.membrane.L)

        fig2 = plt.figure(figsize=figsize)
        plt.xlabel("Passage time (ns)", fontsize=fs)
        plt.ylabel("prob. density", fontsize=fs)
        plt.hist(
            passage_times,
            bins=len(center),
            density=True,
            color=[0, 0.5, 0.5],
            label="prob. density",
        )
        plt.plot(x[1:], y2[1:], label="FPT fit (hom)", color="red", ls="dashed")
        plt.xlim(0, x_lim)
        plt.ylim(0, 1.2 * np.max(histo[0:]))
        plt.xticks(fontsize=fs)
        plt.legend(loc="center right", fontsize=fs)
        plt.tight_layout()

        return fig1, fig2

    def bootstrap_diffusion(self, selector, n_bootstraps, plot=True):
        # TODO implement this function to bootstrap the diffusion coefficient and get statistical insights, e.g. confidence intervals
        raise NotImplementedError

    def permutation_test_diffusion(self, selector):
        # TODO implement this function for further statistical insighs
        raise NotImplementedError

    def store_results_json(self, filename: str = "diffusion_analysis_results"):
        """
        Store the results of the diffusion analysis in a json file.
        Will be saved in the results directory.

        Args:
            filename (str): The name of the json file to store the results in.
        """
        out = {
            "D": self.D,
            "n_passages": self.n_passages,
            "membrane": str(self.membrane),
        }
        # print(out)
        self._store_dict_as_json(out, self.results_dir + filename + ".json")

    def save_passage_times_in_ns_to_txt(self, selector: str, file_name: str):
        """
        Save the passage times in ns to a txt file in the results directory.

        Args:
            selector (str): The selector to save the passage times for.
            file_name (str): The name of the txt file to save the passage times in.
        """
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
        """
        Create a plot of n random passages for the given selector.
        This can be used to randomly validate the passage detection.
        """
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
                x_passages_sel[0],
                y_passages_sel[0],
                z_passages_sel[0],
            )  # starting point
            ax.scatter(
                x_passages_sel[-1],
                y_passages_sel[-1],
                z_passages_sel[-1],
            )  # end point
        return fig, ax

    def plot_z_passages(self, selector: str, pos):
        """
        Plot the z-trajectories of the selected passage pos of the selector.

        Args:
            selector (str): The selector to plot the z-trajectories for.
            pos (int): which one of the passages of the specific selector to plot
        """
        plt.figure()
        index = self.passageIndices[selector][pos]
        start = int(self.passageStarts[selector][pos])
        end = int(start + self.passageTimes[selector][pos])
        start_index = int(start / self.step_size)
        end_index = int(end / self.step_size) + 1
        print(start_index, end_index)
        print(start, end)
        print(self.trajectories[selector][index, start:end, 2])
        print(self.trajectories[selector][index, start:end, 2].shape)
        plt.plot(
            self.trajectories[selector][index, start_index:end_index, 2],
            label=f"Passage {pos}",
        )
        plt.xlabel("Time in ps")
        plt.ylabel("z in A")
        plt.title("Trajectories of the selected passages")
        plt.plot()

    def plot_starting_points(self, selector: str):
        """
        Plot the starting points of the passages for the given selector.
        This is helpful to see if the passages are detected correctly.

        Args:
            selector (str): The selector to plot the starting points for.

        Returns:
            fig (matplotlib.figure.Figure): The figure with the starting points.
        """
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
        # TODO check if the passages are already calculated
        x_passages = self.trajectories[selector][self.passageIndices[selector], :, 0]
        y_passages = self.trajectories[selector][self.passageIndices[selector], :, 1]
        z_passages = self.trajectories[selector][self.passageIndices[selector], :, 2]
        ffs = self.passageStarts[selector] / self.step_size
        ffs = ffs.astype(int)
        scatter = ax.scatter(
            x_passages[np.arange(np.size(x_passages, 0)), ffs + 1] / 10,
            y_passages[np.arange(np.size(x_passages, 0)), ffs + 1] / 10,
            z_passages[np.arange(np.size(x_passages, 0)), ffs + 1] / 10,
            c=z_passages[np.arange(np.size(x_passages, 0)), ffs + 1] / 10,
            cmap="viridis",
        )
        # TODO ugly way of getting the point. maybe there is a better way
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label("z-value (nm)", fontsize="x-large")
        return fig
