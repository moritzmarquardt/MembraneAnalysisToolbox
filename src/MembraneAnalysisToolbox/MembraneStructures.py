from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class Membrane(ABC):
    """
    Abstract class for the membrane structure.
    A membrane is a set of atoms that form a structure in the simulation box of a MD simulation.
    Child classes are meant to implement a specific membrane structure (e.g. hexagonal, cubic).
    All these membrane structures should have the following basic functionality:
    - find_location: Find the location of the membrane in the simulation box and store it as a class attribute
    - print_location: Print the location of the membrane
    - plot_location: Plot the location of the membrane
    - __str__: Return a string representation of the membrane

    Each membrane can have different attributes, depending on the structure.
    The membrane structures are meant to be a collection of attributes and functions that are specific to the structure.
    """

    @abstractmethod
    def find_location(self) -> None:
        pass

    @abstractmethod
    def print_location(self) -> None:
        pass

    @abstractmethod
    def plot_location(self):
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class MembraneForDiffusionAnalysis(Membrane):
    """
    Define what functions are necessary for a membrane to be used in the diffusion analysis.
    """

    @abstractmethod
    def define_isabove_isbelow_funcs(self):
        pass


class MembraneForPoreAnalysis(Membrane):
    """
    Define what functions are necessary for a membrane to be used in the effective pore size analysis.
    """

    @abstractmethod
    def find_zConstraints(self):
        pass

    @abstractmethod
    def find_yConstraints(self):
        pass


class HexagonalMembrane(MembraneForDiffusionAnalysis, MembraneForPoreAnalysis):
    """
    Class for the hexagonal membrane structure.
    It should be able to be used for both the diffusion analysis and the effective pore size analysis.
    A hexagonal membrane is characterised by a lower boundary lowerZ and a thickness L.
    """

    def __init__(
        self,
        selectors: list[str],
        L: float = None,
        lowerZ: float = None,
        isAtomAbove: callable = None,
        isAtomBelow: callable = None,
        y_middle: float = None,
        y_range: float = None,
    ):
        if isinstance(selectors, str):
            selectors = [selectors]
        self.selectors = selectors
        self.L = L
        self.lowerZ = lowerZ
        self.isAtomAbove = isAtomAbove
        self.isAtomBelow = isAtomBelow
        self.y_middle = y_middle
        self.y_range = y_range

    def find_location(self, trajectories):
        # TODO: Use cdf instead of histogram to avoid binning
        """
        Function to find the lower boundary of the hexagonal structure.
        Sets the lowerZ attribute, requires the L attribute.

        params:
            trajectories (np.array): The trajectory of the membrane selector atoms
        """

        if self.L is None:
            raise ValueError("The L attribute is not set.")

        # Get the z-coordinates of the membrane trajectory
        z = trajectories[:, 0, 2].flatten()

        # Calculate the histogram of the z-coordinates
        _, bins = np.histogram(z, bins=100, density=True)

        # Calculate the upper and lower bounds of the histogram and with that the middle value
        z_lower = bins[0]
        z_upper = bins[-1]
        z_middle = (z_lower + z_upper) / 2

        # Calculate the lower boundary of the hexagonal structure and return it
        self.lowerZ = z_middle - self.L / 2

    def print_location(self):
        """
        Print the lower boundary of the hexagonal structure.
        lowerZ attribute must be set (for example with the find_location function).
        """
        if self.lowerZ is None:
            raise ValueError(
                "The lower boundary of the hexagonal structure is not set. Run find_location() first."
            )
        print(f"Lower boundary of the hexagonal structure: {self.lowerZ}")

    def plot_location(self, trajectories):
        """
        Plot the histogram of the z-coordinates of the membrane selector atoms and the boundaries of the hexagonal structure.
        lowerZ attribute must be set (for example with the find_location function).
        """
        if self.lowerZ is None:
            raise ValueError(
                "The lower boundary of the hexagonal structure is not set. Run find_location() first."
            )
        plt.figure()
        z = trajectories[:, 0, 2].flatten()
        plt.hist(z, bins=50, density=True, label="Histogram")
        plt.axvline(self.lowerZ, color="r", linestyle="--", label="Lower boundary")
        plt.axvline(
            self.lowerZ + self.L, color="r", linestyle="--", label="Upper boundary"
        )
        plt.xlabel("z-coordinate")
        plt.ylabel("Density")
        plt.title("Histogram of the z-coordinates of the membrane")
        plt.legend()

    def define_isabove_isbelow_funcs(self):
        """
        Define the isAtomAbove and isAtomBelow functions for the hexagonal structure.
        lowerZ attribute must be set (for example with the find_location function).
        """
        if self.lowerZ is None:
            raise ValueError(
                "The lower boundary of the hexagonal structure is not set. Run find_location() first."
            )

        def is_above(curr):
            # curr is the position to be evaluated
            return curr[2] > self.lowerZ + self.L

        def is_below(curr):
            return curr[2] < self.lowerZ

        self.isAtomAbove = is_above
        self.isAtomBelow = is_below

        return is_above, is_below

    def find_zConstraints(self, ratio=0.8):
        """
        When given all the atom positions of the membrane, this defines the z-constraints for an effective pore size analysis.
        It basically takes the minimum and maximum z-coordinate of the membrane atoms and defines the pore size as ratio (default 80%) of the distance between the two.
        """
        # z_min = membrane_atom_positions[:, :, 2].min()
        # z_max = membrane_atom_positions[:, :, 2].max()
        # z_max = (z_min + z_max) / 2 + (z_max - z_min) / 2 * 0.8
        # z_min = (z_min + z_max) / 2 - (z_max - z_min) / 2 * 0.8
        # return z_min, z_max
        return (
            self.lowerZ + self.L * (1 - ratio) / 2,
            self.lowerZ + self.L * (1 + ratio) / 2,
        )

    def find_yConstraints(self, membrane_atom_positions=None):
        if (self.y_middle or self.y_range) is None:
            raise ValueError("The y_middle and y_range attributes are not set.")
        # y_profile = stats.gaussian_kde(membrane_atom_positions[:,:,1].flatten())
        # plt.figure()
        # x = np.linspace(membrane_atom_positions[:,:,1].min(), membrane_atom_positions[:,:,1].max(), 1000)
        # plt.plot(x, y_profile(x))
        # plt.plot()
        # def holefunc(x):
        #     return x - y_profile(x)
        # y_middle, y_range = 0, 0
        # return y_middle, y_range
        return (self.y_middle, self.y_range)

    def __str__(self) -> str:
        attributes = vars(self)
        attributes_str = ", \n".join(
            f"{key}={value}" for key, value in attributes.items()
        )
        return f"HexagonalMembrane: {attributes_str}"
        # return f"HexagonalMembrane: L={self.L}; selector={self.selector}; lowerZ={self.lowerZ}"


class CubicMembrane(MembraneForDiffusionAnalysis):
    def __init__(
        self,
        selectors: list[str],
        cube_arrangement: tuple,
        cube_size: float,
        pore_radius: float,
        lowerZ: float = None,
        isAtomAbove: callable = None,
        isAtomBelow: callable = None,
    ):
        """
        Args:
            selectors (list[str]): The selector for the membrane atoms
            cube_arrangement (tuple): The arrangement of the cubes in the membrane.
                The tuple should be of the form (n_x, n_y, n_z) where n_x, n_y, n_z are the number of cubes in the x, y, z direction
            cube_size (float): The size of one cube
            pore_radius (float): The radius of the pore
        """
        if isinstance(selectors, str):
            selectors = [selectors]
        self.selectors = selectors
        self.L = cube_arrangement[2] * cube_size
        self.cube_arrangement = cube_arrangement
        self.cube_size = cube_size
        self.pore_radius = pore_radius
        self.lowerZ = lowerZ
        self.isAtomAbove = isAtomAbove
        self.isAtomBelow = isAtomBelow

        c = self.cube_size
        self.middle_points = [
            (i * c + c / 2, j * c + c / 2)
            for i in range(self.cube_arrangement[0])
            for j in range(self.cube_arrangement[1])
        ]

    def find_location(self, trajectories):
        # TODO: Use cdf instead of histogram to avoid binning
        """
        Function to find the lower boundary of the hexagonal structure

        Args:
            trajectories (np.array): The trajectory of the membrane selector atoms
        """

        # Get the z-coordinates of the membrane trajectory
        z = trajectories[:, 0, 2].flatten()

        # Calculate the histogram of the z-coordinates
        _, bins = np.histogram(z, bins=100, density=True)

        # Calculate the upper and lower bounds of the histogram and with that the middle value
        z_lower = bins[0]
        z_upper = bins[-1]

        z_middle = (z_lower + z_upper) / 2

        # Calculate the lower boundary of the cubic structure and return it
        self.lowerZ = z_middle - self.L / 2

    def print_location(self):
        if self.lowerZ is None:
            raise ValueError(
                "The lower boundary of the cubic structure is not set. Run find_location() first."
            )
        print(f"Lower boundary of the cubic structure: {self.lowerZ}")

    def plot_location(self, trajectories):
        if self.lowerZ is None:
            raise ValueError(
                "The lower boundary of the cubic structure is not set. Run find_location() first."
            )
        plt.figure()
        z = trajectories[:, 0, 2].flatten()
        plt.hist(z, bins=100, density=True, label="Histogram")
        plt.axvline(self.lowerZ, color="r", linestyle="--", label="Lower boundary")
        plt.axvline(
            self.lowerZ + self.L, color="r", linestyle="--", label="Upper boundary"
        )
        plt.xlabel("z-coordinate")
        plt.ylabel("Density")
        plt.title("Histogram of the z-coordinates of the membrane")
        plt.legend()

    def define_isabove_isbelow_funcs(self):
        if self.lowerZ is None:
            raise ValueError(
                "The lower boundary of the cubic structure is not set. Run find_location() first."
            )

        def is_above(curr):
            # curr is the position to be evaluated
            lower_border = self.lowerZ + self.L - self.pore_radius
            upper_border = self.lowerZ + self.L
            isAbove = False
            if curr[2] > lower_border:
                isAbove = True
            if curr[2] < upper_border and curr[2] > lower_border:
                dists = [np.sum(np.square(xy - curr[:2])) for xy in self.middle_points]
                min_dist = np.min(dists)
                if min_dist < self.pore_radius**2:
                    isAbove = False

            return isAbove

        def is_below(curr):
            # curr is the position to be evaluated
            lower_border = self.lowerZ
            upper_border = self.lowerZ + self.pore_radius
            isBelow = False
            if curr[2] < upper_border:
                isBelow = True
            if curr[2] > lower_border and curr[2] < upper_border:
                dists = [np.sum(np.square(xy - curr[:2])) for xy in self.middle_points]
                min_dist = np.min(dists)
                if min_dist < self.pore_radius**2:
                    isBelow = False

            return isBelow

        self.isAtomAbove = is_above
        self.isAtomBelow = is_below

        return is_above, is_below

    def calc_passage_length(self, T):
        """
        Function to calculate the passage length of the membrane

        Args:
            T (np.array): The trajectory of the passage (time, axis)
        """
        spheres_crossed = []
        # system 1 is the upper and lower layer with a total of 8 crossings
        spheres_system1_center = [
            [
                i * self.cube_size + self.cube_size / 2,
                j * self.cube_size + self.cube_size / 2,
                l * self.cube_size + self.cube_size / 2,
            ]
            for i in range(self.cube_arrangement[0])
            for j in range(self.cube_arrangement[1])
            for l in range(self.cube_arrangement[2])
        ]
        print(spheres_system1_center)
        # system 2 is the middle layer with a total of 9 crossings
        spheres_system2_center = [
            [i * self.cube_size, j * self.cube_size, self.cube_size]
            for i in range(self.cube_arrangement[0] + 1)
            for j in range(self.cube_arrangement[1] + 1)
        ]
        print(spheres_system2_center)

        system_label = None
        for t in range(T.shape[0]):
            if system_label == 1:
                dist_spheres_system1 = [
                    np.sum(np.square(T[t] - sphere_center))
                    for sphere_center in spheres_system1_center
                ]
                crossed_system1 = [
                    i
                    for i in range(len(dist_spheres_system1))
                    if dist_spheres_system1[i] < self.pore_radius**2
                ]
                if not crossed_system1:
                    system_label = None
                    continue
                if crossed_system1 and spheres_crossed[-1]:
                    spheres_crossed.append(spheres_system1_center[crossed_system1[0]])

            elif system_label == 2:
                pass
            else:
                dist_spheres_system1 = [
                    np.sum(np.square(T[t] - sphere_center))
                    for sphere_center in spheres_system1_center
                ]
                dist_spheres_system2 = [
                    np.sum(np.square(T[t] - sphere_center))
                    for sphere_center in spheres_system2_center
                ]
                crossed_system1 = [
                    i
                    for i in range(len(dist_spheres_system1))
                    if dist_spheres_system1[i] < self.pore_radius**2
                ]
                crossed_system2 = [
                    i
                    for i in range(len(dist_spheres_system2))
                    if dist_spheres_system2[i] < self.pore_radius**2
                ]
                print(crossed_system1, crossed_system2)
                if not crossed_system1 and not crossed_system2:
                    continue
                if crossed_system1:
                    system_label = 1
                    spheres_crossed.append(spheres_system1_center[crossed_system1[0]])
                if crossed_system2:
                    system_label = 2
                    spheres_crossed.append(spheres_system2_center[crossed_system2[0]])

        return spheres_crossed

    def __str__(self) -> str:
        return f"CubicMembrane: L={self.L}; selector={self.selectors}; lowerZ={self.lowerZ}, cube_arrangement={self.cube_arrangement}, cube_size={self.cube_size}"


class Solvent(MembraneForDiffusionAnalysis):
    """
    Membrane Structure implementation to be able to analyse the solvent system, where no membrane is present.
    """

    def __init__(self, lowerZ, upperZ, L):
        self.selectors = None
        self.lowerZ = lowerZ
        self.upperZ = upperZ
        self.L = L
        self.isAtomAbove = None
        self.isAtomBelow = None

    def find_location(self):
        pass

    def print_location(self):
        print(
            f"Solvent system with virtual borders at z={self.lowerZ} and z={self.upperZ}"
        )

    def plot_location(self):
        # TODO plot the distribution of the solvent atoms and the defined borders
        pass

    def define_isabove_isbelow_funcs(self):
        def is_above(curr):
            return curr[2] > self.upperZ

        def is_below(curr):
            return curr[2] < self.lowerZ

        self.isAtomAbove = is_above
        self.isAtomBelow = is_below

        return is_above, is_below

    def __str__(self) -> str:
        return "Solvent: No membrane present"
