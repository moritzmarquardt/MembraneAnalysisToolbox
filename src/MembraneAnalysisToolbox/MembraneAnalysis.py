import os
import sys

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np


class MembraneAnalysis:
    """Parent Class for core functionability of a membrane analysis.

    This is a base class for the development of different membrane analysis routines.
    For example analysing the diffusion or the eff. pore size.
    An analysis needs a traj and topol file of the simulation and a folder where to save the results.

    This class main functionality is the allocation of trajectories using MDAnalysis.
    Additionally it offeres functionability to only allocate every nth frame for efficiency.


    Parameters:
        topology_file (str): The path to the topology file (.tpr) for the simulation.
        trajectory_file (str): The path to the trajectory file (.xtc) for the simulation.
        results_dir (str): The directory to store the analysis results.
        analysis_max_step_size_ps (int, optional):
            The maximum step size in picoseconds for analysis. Defaults to None.
            If it is bigger than the step size of the simulation, only every nth frame is analysed to match the max step size.
            If it is smaller than the simulation step size, the simulation step size will be analysed.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Methods:
        _validate_file_extension(path: str, extension: str) -> bool:
            This is used to check if a file exists and has the correct file extension.

        _allocateTrajectories(selectors):
            Used to allocate the trajectories of the selectors.

        _validate_selectors(selectors) -> list:
            Check if the selectors are a list and if not make them a list.

        _validate_and_create_results_dir(results_dir: str):
            This is used to create a directory to store the results if it does not exist yet.

        create_hist_for_axis(selectors, axis: int) -> tuple:
            Create a histogram to show the distribution of the atoms along an axis.

        _create_histogram(data, label=None, bins=100, title="", xlabel="", ylabel="") -> tuple:
            Create a simple histogram.

        save_fig_to_results(fig, name: str):
            Save a figure to the results directory.

        find_membrane_location_hexstructure(mem_selector: str, L: float) -> float:
            Find the lower boundary of the membrane in a hexagonal structure.

        __str__() -> str:
            Return a string representation of the object.

    Raises:
        FileNotFoundError: If the topology or trajectory file does not exist or has the wrong file format.
        ValueError: If the selectors are not of the correct format.
    """

    def __init__(
        self,
        topology_file: str,
        trajectory_file: str,
        results_dir: str,
        analysis_max_step_size_ps: int = None,
        verbose: bool = True,
    ):
        self.verbose = verbose

        # 1: Check if the topology and trajectory files exist and have the correct file format
        if not (self._validate_file_extension(topology_file, ".tpr")):
            raise FileNotFoundError(
                "Topology file does not exist or has wrong file format:"
                " A .tpr file is required."
            )
        else:
            self.topology_file = topology_file

        if not (self._validate_file_extension(trajectory_file, ".xtc")):
            raise FileNotFoundError(
                "Trajectory file does not exist or has wrong file format:"
                " A .xtc file is required."
            )
        else:
            self.trajectory_file = trajectory_file

        # 2: Check if the results directory exists and create it if it does not
        self._validate_and_create_results_dir(results_dir)
        self.results_dir = results_dir
        if self.verbose:
            print("Results will be saved in: " + self.results_dir + ".")

        self.analysis_max_step_size_ps = analysis_max_step_size_ps

        # 3: Build Universe using the MDAnalysis library
        self.u = mda.Universe(topology_file, trajectory_file)

        # 4: Define how many (nth) frames are being analysed to reduce the amount of data
        if self.analysis_max_step_size_ps is None:
            self.nth_frame = 1
        else:
            self.nth_frame = max(
                1, int(np.floor(self.analysis_max_step_size_ps / self.u.trajectory.dt))
            )

        # the step size of the analysis in ps
        self.step_size = self.u.trajectory.dt * self.nth_frame

        # Calculate the number of analysed frames
        self.n_frames = int(np.ceil(self.u.trajectory.n_frames / self.nth_frame))

        # Trajectories of possible atoms/selectors will be stored in a
        # dictionary for speed and efficiency only trajectories of selectors
        # that are actually needed will be loaded when accesing functions that
        # need them
        self.trajectories = {}

        # Initialize timeline of the analysed frames
        # The timeline is mostly used for plotting purposes
        total_simulation_time = self.step_size * self.n_frames
        self.timeline = np.linspace(0, total_simulation_time, self.n_frames)

    @staticmethod
    def _validate_file_extension(path: str, extension: str) -> bool:
        """This is used to check if a file exists and has the correct file extension.

        Args:
            path (str): The path to the file.
            extension (str): The correct file extension.

        Returns:
            bool: True if the file exists and has the correct file extension, False otherwise.
        """
        return os.path.exists(path) and path.endswith(extension)

    def _allocateTrajectories(self, selectors):
        """Used to allocate the trajectories of the selectors.

        The selectors define the atoms of the MDAnalysis Universe that have to be allocated.
        The trajectories of these atoms are stored in a dictionary for speed and efficiency.
        Only every self.nth frame is stored to reduce the amount of data.

        Args:
            selectors (str or list): The selectors to allocate the trajectories for.

        Raises:
            ValueError: If the selectors are not of the correct format.

        Returns:
            None

        """
        selectors = self._validate_selectors(selectors)

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

    @staticmethod
    def _validate_selectors(selectors) -> list:
        """
        Check if the selectors are a list and if not make them a list.

        Args:
            selectors (str or list): The selectors to validate.

        Returns:
            list: The selectors as a
        """
        if isinstance(selectors, str):
            selectors = [selectors]
        if not (isinstance(selectors, list) or isinstance(selectors, str)):
            raise ValueError("Selectors must be a string or list of strings.")
        return selectors

    def _validate_and_create_results_dir(self, results_dir: str):
        """
        This is used to create a directory to store the results if it does not exist yet.

        Args:
            results_dir (str): The directory to store the results.

        Raises:
            Exception: If the results directory does not end with a '/'.
        """
        if not results_dir.endswith("/"):
            raise Exception(
                f"The results directory: {results_dir} must end with a '/'."
            )

        # Warn if the results directory is not in the same directory as the trajectory file
        if not os.path.dirname(self.trajectory_file) in os.path.dirname(results_dir):
            print(
                "Warning: The results directory is not in the same directory as the trajectory file."
            )

        if not os.path.isdir(results_dir):
            if self.verbose:
                print("Creating results directory: " + results_dir + ".")
            os.makedirs(results_dir)

    def create_hist_for_axis(self, selectors, axis: int) -> tuple:
        """Create a histogram to show the distribution of the atoms along an axis.

        This can be used to see the distribution of the atoms along the x, y, or z-axis.

        Args:
            selectors (list): The selectors to allocate the trajectories for.
            axis (int): The axis to create the histogram for (0=x, 1=y, 2=z).

        """
        selectors = self._validate_selectors(selectors)
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
            title="Histogram of axis" + str(axis),
            xlabel="axis" + str(axis) + " [Angstrom]",
            ylabel="Frequency",
        )
        ax_hist.legend()
        return fig_hist, ax_hist

    @staticmethod
    def _create_histogram(
        data, label=None, bins=100, title="", xlabel="", ylabel=""
    ) -> tuple:
        """Create a simple histogram.

        This exists to make it easy to have all histos in the same style

        Args:
            data (array): The data to create the histogram for.
            label (str, optional): The label for the histogram. Defaults to None.
            bins (int, optional): The number of bins for the histogram. Defaults to 100.
            title (str, optional): The title of the histogram. Defaults to "".
            xlabel (str, optional): The x-axis label of the histogram. Defaults to "".
            ylabel (str, optional): The y-axis label of the histogram. Defaults to "".

        Returns:
            tuple: The figure and axis of the histogram.
        """
        fig_hist, ax_hist = plt.subplots()
        fig_hist.suptitle(title, fontsize="x-large")
        if label is not None:
            ax_hist.hist(data, bins=bins, density=True, alpha=0.5, label=label)
        else:
            ax_hist.hist(data, bins=bins, density=True, alpha=0.5)
        if xlabel is not None:
            ax_hist.set_xlabel(xlabel, fontsize="x-large")
        if ylabel is not None:
            ax_hist.set_ylabel(ylabel, fontsize="x-large")
        ax_hist.legend()

        return fig_hist, ax_hist

    def save_fig_to_results(self, fig, name: str):
        """Save a figure to the results directory.

        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            name (str): The name of the figure.

        """
        fig.savefig(self.results_dir + name + ".png")
        if self.verbose:
            print("Figure saved in: " + self.results_dir + name + ".png")

    # TODO: instead of binning use CDF
    def find_membrane_location_hexstructure(self, mem_selector: str, L: float) -> float:
        """
        Find the lower boundary of the membrane in a hexagonal structure.

        Args:
            mem_selector (str): The selector for the membrane atoms.
            L (float): The length of the hexagonal structure.

        Returns:
            float: The lower boundary of the membrane in the hexagonal structure.
        """
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
