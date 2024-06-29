import os
import sys

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np


class MembraneAnalysis:
    """
    This is a parent class for basic membrane analysis.
    Idea is that this is a base class for the development of different
    membrane analysis routines. Like analysing the diffusion or the
    eff. pore size.

    """

    def __init__(
        self,
        topology_file: str,
        trajectory_file: str,
        analysed_max_step_size_ps: int = None,
        results_dir: str = None,
        verbose: bool = True,
    ):
        self.verbose = verbose

        if not (self._check_existence_and_format(topology_file, ".tpr")):
            raise FileNotFoundError(
                "Topology file does not exist or has wrong file format:"
                " A .tpr file is required."
            )
        else:
            self.topology_file = topology_file

        if not (self._check_existence_and_format(trajectory_file, ".xtc")):
            raise FileNotFoundError(
                "Trajectory file does not exist or has wrong file format:"
                " A .xtc file is required."
            )
        else:
            self.trajectory_file = trajectory_file

        self.analysed_max_step_size_ps = analysed_max_step_size_ps

        self.results_dir = self._pop_results_dir(results_dir)

        # Build Universe using the MDAnalysis library
        self.u = mda.Universe(topology_file, trajectory_file)

        # Define how many (nth) frames are being analysed to reduce the amount
        # of data
        if self.analysed_max_step_size_ps is None:
            self.nth_frame = 1
        else:
            self.nth_frame = int(
                np.floor(self.analysed_max_step_size_ps / self.u.trajectory.dt)
            )

        # the step size of the analysis in ps
        self.step_size = self.u.trajectory.dt * self.nth_frame

        # Calculate the number of analysed frames
        self.n_frames = int(np.ceil(self.u.trajectory.n_frames / self.nth_frame))

        self.trajectory_npz_file_path = (
            str(self.results_dir)
            + "analysis_trajectories"
            + "_nth"
            + str(self.nth_frame)
            + "_dt"
            + str(self.step_size)
            + "_nframes"
            + str(self.n_frames)
            + ".npz"
        )

        # Trajectories of possible atoms/selectors will be stored in a
        # dictionary for speed and efficiency only trajectories of selectors
        # that are actually needed will be loaded when accesing functions that
        # need them
        self.trajectories = {}
        # self.load_trajectories_if_possible()

        # Initialize timeline of the analysed frames
        # The timeline is mostly used for plotting purposes
        total_simulation_time = self.step_size * self.n_frames
        self.timeline = np.linspace(0, total_simulation_time, self.n_frames)

    @staticmethod
    def _check_existence_and_format(path, extension):
        return os.path.exists(path) and path.endswith(extension)

    def _allocateTrajectories(self, selectors):
        # check for the format of the selectors
        if isinstance(selectors, str):
            selectors = [selectors]
        if not (isinstance(selectors, list) or isinstance(selectors, str)):
            raise ValueError("Selectors must be a string or list of strings.")

        sels_unstored = [s for s in selectors if s not in self.trajectories]
        if len(sels_unstored) > 0:
            if self.verbose:
                print(
                    'Allocating trajectories for selectors: "'
                    + '", "'.join(sels_unstored)
                    + '".'
                )
            atomslist = [self.u.select_atoms(sel) for sel in sels_unstored]
            positions = np.zeros(
                (sum([atoms.n_atoms for atoms in atomslist]), self.n_frames, 3)
            )
            indexes = [0]
            for atoms in atomslist:
                indexes.append(indexes[-1] + atoms.n_atoms)
            for i, _ in enumerate(self.u.trajectory[:: self.nth_frame]):
                if self.verbose:
                    percentage = int((i + 1) / self.n_frames * 100)
                    sys.stdout.write(f"\r\tProgress: {percentage}%")
                    sys.stdout.flush()
                for j, atoms in enumerate(atomslist):
                    positions[indexes[j] : indexes[j + 1], i, :] = atoms.positions
            for i, sele in enumerate(sels_unstored):
                self.trajectories[sele] = positions[indexes[i] : indexes[i + 1], :, :]

            if self.verbose:
                print("\nTrajectories allocated.")

    def _pop_results_dir(self, results_dir=None):
        # Create a directory to store the results if it is not given or
        # already exists
        if isinstance(results_dir, str):
            if os.path.isdir(results_dir):
                return results_dir
            else:
                self._create_directory(results_dir)
                return results_dir
        elif results_dir is None:
            path = os.path.dirname(self.trajectory_file) + "/"
            results_dir = path + "analysis/"
            if os.path.isdir(results_dir):
                return results_dir
            else:
                self._create_directory(results_dir)
                return results_dir
        if self.verbose:
            print("Results will be saved in: " + self.results_dir + ".")

    def _create_directory(self, results_dir):
        if self.verbose:
            print("Creating results directory: " + results_dir + ".")
        os.makedirs(results_dir)

    @staticmethod
    def _create_histogram(data, label=None, title="", xlabel=None, ylabel=None):
        # this exists to make it easy to have all histos in the same style
        fig_hist, ax_hist = plt.subplots()
        fig_hist.suptitle(title, fontsize="x-large")
        if label is not None:
            ax_hist.hist(data, bins=100, density=True, alpha=0.5, label=label)
        else:
            ax_hist.hist(data, bins=100, density=True, alpha=0.5)
        if xlabel is not None:
            ax_hist.set_xlabel(xlabel, fontsize="x-large")
        if ylabel is not None:
            ax_hist.set_ylabel(ylabel, fontsize="x-large")
        ax_hist.legend()

        return fig_hist, ax_hist

    def create_hist_for_axis(self, selectors: list, axis: int):
        self._allocateTrajectories(selectors)
        total_elements = sum(
            self.trajectories[sele].shape[0] * self.trajectories[sele].shape[1]
            for sele in selectors
        )
        x = np.empty(total_elements)
        index = 0
        for sele in selectors:
            positions = self.trajectories[sele]
            n_elements = positions.shape[0] * positions.shape[1]
            x[index : index + n_elements] = positions[:, :, axis].flatten()
            index += n_elements

        fig_hist, ax_hist = self._create_histogram(
            data=x,
            label=selectors,
            title="Histogram of x",
            xlabel="x",
            ylabel="Frequency",
        )
        return fig_hist, ax_hist

    def save_fig_to_results(self, fig, name=None):
        fig.savefig(self.results_dir + name + ".png")
        if self.verbose:
            print("Figure saved in: " + self.results_dir + name + ".png")

    def save_trajectories_if_notthere(self):
        if self.verbose:
            print("Saving trajectories...")
        if not os.path.exists(self.trajectory_npz_file_path):
            self._save_trajectories()

    def _save_trajectories(self):
        np.savez_compressed(self.trajectory_npz_file_path, **self.trajectories)
        if self.verbose:
            print("Trajectories saved in: " + self.trajectory_npz_file_path)

    def load_trajectories_if_possible(self):
        # TODO add a check to see if the file is corrupted
        if os.path.exists(self.trajectory_npz_file_path):
            if self.verbose:
                print("Loading trajectories from file...")
            self._load_trajectories()

    def _load_trajectories(self):
        with np.load(self.trajectory_npz_file_path) as data:
            for key in data.keys():
                self.trajectories[key] = data[key]

    def find_membrane_location_hexstructure(self, mem_selector: str, L):
        self._allocateTrajectories(mem_selector)

        # Get the z-coordinates of the membrane trajectory
        z = self.trajectories[mem_selector][:, :, 2].flatten()

        # Calculate the histogram of the z-coordinates
        _, bins = np.histogram(z, bins=100, density=True)

        # Calculate the upper and lower bounds of the histogram and with that the middle value
        z_lower = bins[0]
        z_upper = bins[-1]
        z_middle = (z_lower + z_upper) / 2

        # Calculate the lower boundary of the hexagonal structure and return it
        return z_middle - L / 2

    def __str__(self) -> str:
        return f"MembraneAnalysis object with {self.n_frames} frames analysed."
