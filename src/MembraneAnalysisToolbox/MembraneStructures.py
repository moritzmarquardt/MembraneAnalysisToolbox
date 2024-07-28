from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class Membrane(ABC):
    @abstractmethod
    def find_location(self):
        pass

    def print_location(self):
        pass

    def plot_location(self):
        pass

    @abstractmethod
    def define_isabove_isbelow_funcs(self):
        pass


class HexagonalMembrane(Membrane):
    def __init__(self, selector, L):
        self.selector = selector
        self.L = L
        self.lowerZ = None
        self.isAtomAbove = None
        self.isAtomBelow = None

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

        # Calculate the lower boundary of the hexagonal structure and return it
        self.lowerZ = z_middle - self.L / 2

    def print_location(self):
        print(f"Lower boundary of the hexagonal structure: {self.lowerZ}")

    def plot_location(self, trajectories):
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
        def is_above(curr):
            # curr is the position to be evaluated
            return curr[2] > self.lowerZ + self.L

        def is_below(curr):
            return curr[2] < self.lowerZ

        self.isAtomAbove = is_above
        self.isAtomBelow = is_below

        return is_above, is_below

    def __str__(self) -> str:
        return f"HexagonalMembrane: L={self.L}; selector={self.selector}; lowerZ={self.lowerZ}"


class CubicMembrane(Membrane):
    def __init__(
        self,
        selector: str,
        cube_arrangement: tuple,
        cube_size: float,
        pore_radius: float,
    ):
        """
        Args:
            selector (str): The selector for the membrane atoms
            cube_arrangement (tuple): The arrangement of the cubes in the membrane.
                The tuple should be of the form (n_x, n_y, n_z) where n_x, n_y, n_z are the number of cubes in the x, y, z direction
            cube_size (float): The size of one cube
        """
        self.selector = selector
        self.L = cube_arrangement[2] * cube_size
        self.lowerZ = None
        self.cube_arrangement = cube_arrangement
        self.cube_size = cube_size
        self.pore_radius = pore_radius
        self.isAtomAbove = None
        self.isAtomBelow = None

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

        # Calculate the lower boundary of the hexagonal structure and return it
        self.lowerZ = z_middle - self.L / 2

    def print_location(self):
        print(f"Lower boundary of the hexagonal structure: {self.lowerZ}")

    def plot_location(self, trajectories):
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

    def __str__(self) -> str:
        return f"HexagonalMembrane: L={self.L}; selector={self.selector}; lowerZ={self.lowerZ}, cube_arrangement={self.cube_arrangement}, cube_size={self.cube_size}"
