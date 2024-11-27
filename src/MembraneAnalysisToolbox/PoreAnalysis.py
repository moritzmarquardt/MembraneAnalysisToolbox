from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
import skimage as ski
from scipy.stats import gaussian_kde

from MembraneAnalysisToolbox.MembraneAnalysis import MembraneAnalysis
from MembraneAnalysisToolbox.MembraneStructures import MembraneForPoreAnalysis


class PoreAnalysis(MembraneAnalysis):
    """
    Class for analyzing Membrane pores.
    for example the effective pore size or a density plot of the membrane pore.
    """

    def __init__(
        self,
        topology_file: str,
        trajectory_file: str,
        membrane: MembraneForPoreAnalysis,
        analysis_max_step_size_ps: int = None,
        results_dir: str = None,
        verbose=False,
    ):
        if not isinstance(membrane, MembraneForPoreAnalysis):
            raise ValueError(
                "The membrane object must be an instance of the abstract MembraneForPoreAnalysis."
            )
        super().__init__(
            topology_file=topology_file,
            trajectory_file=trajectory_file,
            analysis_max_step_size_ps=analysis_max_step_size_ps,
            results_dir=results_dir,
            verbose=verbose,
            membrane=membrane,
        )

    def analyseDensity(
        self,
        selectors,
        z_range,
        skip=50,
        bw="scott",
    ):
        self._allocateTrajectories(selectors)

        selectors_positions = []
        for selector in selectors:
            selectors_positions.append(self.trajectories[selector][:, ::skip, :])
        selectors_positions = np.concatenate(selectors_positions, axis=0)
        print(selectors_positions.shape)

        mask = (selectors_positions[:, :, 2] >= z_range[0]) & (
            selectors_positions[:, :, 2] <= z_range[1]
        )
        filtered_positions = selectors_positions[mask][:, 0:2]
        print(filtered_positions.shape)

        kde = gaussian_kde(filtered_positions.T, bw_method=bw)

        print("Kde faktoren (mem, solv): " + str(kde.factor))

        # Evaluate the KDE on a grid and normalise it
        xmin = min(filtered_positions[:, 0])
        xmax = max(filtered_positions[:, 0])
        xn = int((xmax - xmin))
        ymin = min(filtered_positions[:, 1])
        ymax = max(filtered_positions[:, 1])
        yn = int((ymax - ymin))
        x = np.linspace(xmin, xmax, xn)
        y = np.linspace(ymin, ymax, yn)
        X, Y = np.meshgrid(x, y)
        Z = np.reshape(kde([X.ravel(), Y.ravel()]), X.shape)
        plt.figure()
        plt.title(
            f"Density of {selectors} in the Z-range {int(z_range[0])} to {int(z_range[1])}",
            fontsize="x-large",
        )
        plt.xlabel("X-axis in Angstrom", fontsize="x-large")
        plt.ylabel("Y-axis in Angstrom", fontsize="x-large")
        plt.pcolormesh(X, Y, Z, shading="gouraud")
        plt.colorbar()
        plt.axis("equal")

    def analyseDensityNormed(
        self,
        selectors,
        z_range,
        skip=50,
        bw="scott",
        plot=True,
    ):
        """same thing as analyseDensity but each selector is normalised on its own and added to the plot"""
        self._allocateTrajectories(selectors)

        selectors_positions = []
        for selector in selectors:
            selectors_positions.append(self.trajectories[selector][:, ::skip, :])

        filtered_positions_list = []
        for sp in selectors_positions:
            mask = (sp[:, :, 2] >= z_range[0]) & (sp[:, :, 2] <= z_range[1])
            filtered_positions = sp[mask][:, 0:2]
            print(filtered_positions.shape)
            filtered_positions_list.append(filtered_positions)

        xmin = min([min(fp[:, 0]) for fp in filtered_positions_list])
        xmax = max([max(fp[:, 0]) for fp in filtered_positions_list])
        xn = int((xmax - xmin))
        ymin = min([min(fp[:, 1]) for fp in filtered_positions_list])
        ymax = max([max(fp[:, 1]) for fp in filtered_positions_list])
        yn = int((ymax - ymin))
        x = np.linspace(xmin, xmax, xn)
        y = np.linspace(ymin, ymax, yn)
        X, Y = np.meshgrid(x, y)

        Z_list = []
        for filtered_positions in filtered_positions_list:
            kde = gaussian_kde(filtered_positions.T, bw_method=bw)
            print("Kde faktor: " + str(kde.factor))
            Z = np.reshape(kde([X.ravel(), Y.ravel()]), X.shape)
            Z_list.append(Z)

        # norm all Z and add them and plot it
        max_Z = [np.max(Z) for Z in Z_list]
        print(max_Z)
        Z_list = [Z / max_Z[i] for i, Z in enumerate(Z_list)]
        Z = sum(Z_list)

        if plot:
            plt.figure()
            plt.title(
                f"Density of {selectors} in the Z-range {int(z_range[0])} to {int(z_range[1])}",
                fontsize="x-large",
            )
            plt.xlabel("X-axis in Angstrom", fontsize="x-large")
            plt.ylabel("Y-axis in Angstrom", fontsize="x-large")
            plt.pcolormesh(X, Y, Z, shading="gouraud")
            plt.colorbar()
            plt.axis("equal")

        return Z

    def calculateEffectivePoreSize(
        self,
        solvent_selectors: list[str],
        z_constraints: Tuple[float, float],
        y_constraints: Tuple[float, float],
        strategy: str,
        bins="auto",
        skip: int = 50,
    ):
        self._allocateTrajectories(solvent_selectors)
        self._allocateTrajectories(self.membrane.selectors)

        combined_membrane_trajectories = []
        for selector in self.membrane.selectors:
            combined_membrane_trajectories.append(
                self.trajectories[selector][:, ::skip, :]
            )
        combined_membrane_trajectories = np.concatenate(combined_membrane_trajectories)

        combined_solvent_trajectories = []
        for selector in solvent_selectors:
            combined_solvent_trajectories.append(
                self.trajectories[selector][:, ::skip, :]
            )
        combined_solvent_trajectories = np.concatenate(combined_solvent_trajectories)

        membrane_atom_positions_filtered = self._filterPositions(
            positions=combined_membrane_trajectories,
            y_constraints=y_constraints,
            z_constraints=z_constraints,
        )
        solvent_atom_positions_filtered = self._filterPositions(
            positions=combined_solvent_trajectories,
            y_constraints=y_constraints,
            z_constraints=z_constraints,
        )

        bins = bins
        membrane_hist, membrane_hist_edges = np.histogram(
            membrane_atom_positions_filtered, density=1, bins=bins
        )
        solvent_hist, solvent_hist_edges = np.histogram(
            solvent_atom_positions_filtered, density=1, bins=bins
        )

        pore_edges = self._calculate_pore_edges(
            membrane_hist,
            membrane_hist_edges,
            solvent_hist,
            solvent_hist_edges,
            strategy=strategy,
        )

        self.plotEffectivePoreSize(
            membrane_hist,
            membrane_hist_edges,
            solvent_hist,
            solvent_hist_edges,
            pore_edges,
        )

        return (pore_edges[0], pore_edges[1])

    def findPoreCenter(
        self,
        z_constraints: Tuple[float, float],
        x_threshold: Tuple[float, float],
        y_threshold: Tuple[float, float],
        radius: float,
        skip: int = 50,
        bw="scott",
    ):
        combined_membrane_trajectories = []
        for selector in self.membrane.selectors:
            combined_membrane_trajectories.append(
                self.trajectories[selector][:, ::skip, :]
            )
        combined_membrane_trajectories = np.concatenate(combined_membrane_trajectories)

        mask = (
            (combined_membrane_trajectories[:, :, 2] >= z_constraints[0])
            & (combined_membrane_trajectories[:, :, 2] <= z_constraints[1])
            & (combined_membrane_trajectories[:, :, 0] >= x_threshold[0])
            & (combined_membrane_trajectories[:, :, 0] <= x_threshold[1])
            & (combined_membrane_trajectories[:, :, 1] >= y_threshold[0])
            & (combined_membrane_trajectories[:, :, 1] <= y_threshold[1])
        )
        filtered_positions = combined_membrane_trajectories[mask][:, 0:2]

        xmin = min(filtered_positions[:, 0])
        xmax = max(filtered_positions[:, 0])
        ymin = min(filtered_positions[:, 1])
        ymax = max(filtered_positions[:, 1])
        xn = int((xmax - xmin))
        yn = int((ymax - ymin))
        x = np.linspace(xmin, xmax, xn)
        y = np.linspace(ymin, ymax, yn)
        X, Y = np.meshgrid(x, y)

        kde = gaussian_kde(filtered_positions.T, bw_method=bw)
        Z = np.reshape(kde([X.ravel(), Y.ravel()]), X.shape)
        Z = Z / np.max(Z)

        def circle_area_integral(m_x, m_y, r):
            # mask after distance to center
            rows, cols = Z.shape
            # Create a grid of coordinates relative to (x, y)
            y_grid, x_grid = np.ogrid[:rows, :cols]
            # Calculate squared distance from (x, y)
            dist_squared = (x_grid - m_x) ** 2 + (y_grid - m_y) ** 2
            # Create a mask for points within radius r
            circular_mask = dist_squared <= r**2
            # Sum the values in the circle
            sum = np.sum(np.ma.masked_where(~circular_mask, Z))

            return sum

        # Define the function to be minimized
        def objective(params):
            m_x, m_y = params
            return circle_area_integral(m_x, m_y, radius)

        print("lets minimize")
        # Perform the optimization, dual annealing is a global optimization algorithm
        bounds = [
            (0, Z.shape[1]),
            (0, Z.shape[0]),
        ]  # indexing is reverted because of the way the grid is created
        print(bounds)
        result = optimize.dual_annealing(objective, bounds=bounds)
        print(result)

        # Extract the optimal parameters. reverted indexing because of the way the grid is created (in an image science way)
        m_y_opt, m_x_opt = result.x

        print("Optimal m_x:", m_x_opt)
        print("Optimal m_y:", m_y_opt)

        # Plot the histogram and optimal circle
        plt.figure()
        plt.imshow(Z, origin="lower")
        plt.axis("equal")
        plt.colorbar()
        circle = plt.Circle((m_y_opt, m_x_opt), radius, color="r", fill=False)
        plt.gca().add_artist(circle)
        plt.show()

        return m_x_opt + x_threshold[0], m_y_opt + y_threshold[0]

    def radialDensityFunction(self, resnames):
        pass

    def analyseConstraints(self, membrane_selector, y_constraints, z_constraints):
        """
        Analyze the constraints on the atom positions.

        Parameters:
        - y_min (float): Minimum value for the y-axis constraint.
        - y_max (float): Maximum value for the y-axis constraint.
        - z_min (float): Minimum value for the z-axis constraint.
        - z_max (float): Maximum value for the z-axis constraint.
        """
        y_min, y_max = y_constraints
        z_min, z_max = z_constraints
        membrane_atom_positions = self.trajectories[membrane_selector]
        plt.figure()
        plt.hist(membrane_atom_positions[:, :, 0].flatten())
        plt.title("Histogram for x-axis")
        plt.xlabel("X-axis in Angstroms")
        plt.ylabel("Frequency")

        plt.figure()
        plt.hist(membrane_atom_positions[:, :, 1].flatten(), bins=100)
        # plt.legend()
        plt.axvline(x=y_min, color="r", linestyle="--")  # y_min line
        plt.axvline(x=y_max, color="r", linestyle="--")  # y_max line
        plt.title("Histogram for y-axis")
        plt.xlabel("X-axis in Angstroms")
        plt.ylabel("Frequency")

        plt.figure()
        plt.hist(membrane_atom_positions[:, :, 2].flatten(), bins=100)
        plt.axvline(x=z_min, color="r", linestyle="--")  # z_min line
        plt.axvline(x=z_max, color="r", linestyle="--")  # z_max line
        plt.title("Histogram for z-axis")
        plt.xlabel("X-axis in Angstroms")
        plt.ylabel("Frequency")
        # plt.show()

    def _calculate_pore_edges(
        self,
        membrane_hist,
        membrane_hist_edges,
        solvent_hist,
        solvent_hist_edges,
        strategy="averaging",
    ):
        """
        Calculate the lower and upper edges of the pore size distribution based on the averaging mathod.

        Parameters:
        c_hist (numpy.ndarray): Histogram of the concentration values.
        c_bin_edges (numpy.ndarray): Bin edges of the concentration histogram.
        dod_hex_hist (numpy.ndarray): Histogram of the DOD hex values.
        dod_hex_bin_edges (numpy.ndarray): Bin edges of the DOD hex histogram.

        Returns:
        tuple: A tuple containing the average lower edge and average upper edge of the pore size distribution.
        """
        if strategy == "averaging":
            first_zero_bin = np.where(membrane_hist == 0)[0][0]
            first_zero_middle = (
                membrane_hist_edges[first_zero_bin]
                + membrane_hist_edges[first_zero_bin + 1]
            ) / 2
            first_non_zero_bin = np.where(solvent_hist > 0)[0][0]
            first_non_zero_middle = (
                solvent_hist_edges[first_non_zero_bin]
                + solvent_hist_edges[first_non_zero_bin + 1]
            ) / 2
            avrg_lower_edge = np.abs(first_zero_middle + first_non_zero_middle) / 2

            last_zero_bin = np.where(membrane_hist == 0)[0][-1]
            last_zero_middle = (
                membrane_hist_edges[last_zero_bin]
                + membrane_hist_edges[last_zero_bin + 1]
            ) / 2
            last_non_zero_bin = np.where(solvent_hist > 0)[0][-1]
            last_non_zero_middle = (
                solvent_hist_edges[last_non_zero_bin]
                + solvent_hist_edges[last_non_zero_bin + 1]
            ) / 2
            avrg_upper_edge = np.abs(last_zero_middle + last_non_zero_middle) / 2

        elif strategy == "intersection":
            """
            calculate pore size boarders by calculating the intersection of the two histograms
            """
            # Find the last intersection point of the histograms

            def intersection(x):
                return self.hist_linear_interpol_eval(
                    x, membrane_hist_edges, membrane_hist
                ) - self.hist_linear_interpol_eval(x, solvent_hist_edges, solvent_hist)

            min_x_membrane = membrane_hist_edges[0]
            max_x_membrane = membrane_hist_edges[-1]
            # print(min_x_membrane, max_x_membrane)
            avrg_lower_edge = optimize.root_scalar(
                intersection,
                bracket=[min_x_membrane, (max_x_membrane + min_x_membrane) / 2],
                method="brentq",
            ).root
            avrg_upper_edge = optimize.root_scalar(
                intersection,
                bracket=[(max_x_membrane + min_x_membrane) / 2, max_x_membrane],
                method="brentq",
            ).root

        return avrg_lower_edge, avrg_upper_edge

    def plotEffectivePoreSize(
        self,
        membrane_hist,
        membrane_hist_edges,
        solvent_hist,
        solvent_hist_edges,
        edges,
    ):
        """
        Plot the data.

        Returns:
        - None
        """
        lower_edge, upper_edge = edges

        plt.figure()
        x = np.linspace(membrane_hist_edges[0], membrane_hist_edges[-1], 1000)
        plt.plot(
            x / 10,
            self.hist_linear_interpol_eval(x, membrane_hist_edges, membrane_hist),
            label="C",
        )
        plt.plot(
            x / 10,
            self.hist_linear_interpol_eval(x, solvent_hist_edges, solvent_hist),
            label="HEX & DOD",
        )
        plt.axvline(x=lower_edge / 10, color="r", linestyle="--")
        plt.axvline(x=upper_edge / 10, color="r", linestyle="--")
        plt.xlabel("X-axis in nm", fontsize="x-large")
        plt.ylabel("Frequency", fontsize="x-large")
        plt.title(
            "Histogram along the X-axis with Z and Y constraints", fontsize="x-large"
        )
        plt.grid(True)
        plt.legend(fontsize="large")
        # plt.show()

    @staticmethod
    def hist_linear_interpol_eval(x, hist_edges, hist):
        """
        Evaluate the histogram function at a given x value.
        The histogram function is a piecewise linear interpolation of the histogram.
        """
        hist_edges_middle = hist_edges[0:-1] + np.diff(hist_edges) / 2
        front_edge = hist_edges_middle[0] * 2 - hist_edges_middle[1]
        back_edge = hist_edges_middle[-1] * 2 - hist_edges_middle[-2]
        hist_edges_middle = np.append(
            front_edge, np.append(hist_edges_middle, back_edge)
        )
        hist = np.append(0, np.append(hist, 0))
        return np.interp(x, hist_edges_middle, hist)

    @staticmethod
    def _filterPositions(positions, y_constraints, z_constraints):
        """
        Filter positions based on y and z constraints.

        Parameters:
        positions (ndarray): Array of positions with shape (n_atoms, n_timesteps, 3).
        y_constraints (Tuple): Tuple containing the y-axis constraints (y_min, y_max).
        z_constraints (Tuple): Tuple containing the z-axis constraints (z_min, z_max).

        Returns:
        ndarray: Array of filtered x positions.
        """
        y_min, y_max = y_constraints
        z_min, z_max = z_constraints
        mask = (
            (positions[:, :, 1] >= y_min)
            & (positions[:, :, 1] <= y_max)
            & (positions[:, :, 2] >= z_min)
            & (positions[:, :, 2] <= z_max)
        )
        filtered_x_positions = positions[mask, 0].flatten()
        return filtered_x_positions
