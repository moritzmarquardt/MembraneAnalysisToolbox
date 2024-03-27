import MDAnalysis as mda
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import time

class EffectivePoreSizeAnalysis:
    """
    Class for analyzing effective pore size.
    Actually, the analysis is just a sequence of method calls. But here it is encapsulated in a class for clarity.

    Attributes:
    - topology_file (str): The path to the topology file: either .gro or .tpr.
    - trajectory_file (str, optional): The path to the trajectory file: .xtc file. Defaults to None.
    - membrane_resnames (list, optional): List of residue names for the membrane atoms. Defaults to None.
    - solvent_resnames (list, optional): List of residue names for the solvent atoms. Defaults to None.
    - y_min (float): Minimum value for the y-axis constraint.
    - y_max (float): Maximum value for the y-axis constraint.
    - z_min (float): Minimum value for the z-axis constraint.
    - z_max (float): Maximum value for the z-axis constraint.
    """

    def __init__(
            self, 
            topology_file: str, 
            trajectory_file: str = None,
            membrane_resnames = None,
            solvent_resnames = None,
            y_middle = None, 
            y_range = None, 
            verbose = False
            ):
        """
        Initialize the EffectivePoreSizeAnalysis class.

        Parameters:
        - topology_file (str): The path to the topology file: either .gro or .tpr.
        - trajectory_file (str, optional): The path to the trajectory file: .xtc file. Defaults to None.
        - membrane_resnames (list, optional): List of residue names for the membrane atoms. Defaults to None.
        - solvent_resnames (list, optional): List of residue names for the solvent atoms. Defaults to None.
        - y_min (float): Minimum value for the y-axis constraint.
        - y_max (float): Maximum value for the y-axis constraint.
        - z_min (float): Minimum value for the z-axis constraint.
        - z_max (float): Maximum value for the z-axis constraint.
        """

        self.start = time.time()
        
        if os.path.splitext(topology_file)[1] == '.tpr' and trajectory_file is None:
            raise ValueError("A trajectory file is required when using a .tpr topology file.")
        if not (os.path.splitext(topology_file)[1] == '.tpr' or os.path.splitext(topology_file)[1] == '.gro'):
            raise ValueError("A topology file is required. Either a gro or a tpr file.")
        self.verbose = verbose
        self.y_middle = y_middle
        self.y_range = y_range
        
        self.Universe = mda.Universe(topology_file, trajectory_file)
        if self.verbose:
            print("Universe: " + str(self.Universe))
            print("timesteps: " + str(self.Universe.trajectory.n_frames))

        self.membrane_atom_positions = self._readPositions(membrane_resnames)
        self.solvent_atom_positions = self._readPositions(solvent_resnames)
        if self.verbose:
            print("Positions read after " + str(time.time() - self.start) + " seconds")

        self.z_min, self.z_max = self._find_zConstraints(self.membrane_atom_positions) # find the z constraints self.z_min and self.z_max
        # self.z_min = 133
        # self.z_max = 143
        self.y_middle, self.y_range = self._find_yConstraints(self.membrane_atom_positions) # find the y constraints self.y_min and self.y_max

        self.membrane_atom_positions_filtered = self._filterPositions(self.membrane_atom_positions, self.y_middle, self.y_range, self.z_min, self.z_max)
        self.solvent_atom_positions_filtered = self._filterPositions(self.solvent_atom_positions, self.y_middle, self.y_range, self.z_min, self.z_max)
        if self.verbose:
            print("Positions filtered after " + str(time.time() - self.start) + " seconds")

        bins = "auto"
        self.membrane_hist, self.membrane_hist_edges = np.histogram(self.membrane_atom_positions_filtered, density=1,bins=bins)
        self.solvent_hist, self.solvent_hist_edges = np.histogram(self.solvent_atom_positions_filtered, density=1,bins=bins)
        if self.verbose:
            print("Histograms calculated after " + str(time.time() - self.start) + " seconds")

        def membrane_hist_func(x):
            membrane_hist_edges_middle = self.membrane_hist_edges[0:-1] + np.diff(self.membrane_hist_edges)/2
            front_edge = membrane_hist_edges_middle[0] * 2 - membrane_hist_edges_middle[1]
            back_edge = membrane_hist_edges_middle[-1] * 2 - membrane_hist_edges_middle[-2]
            membrane_hist_edges_middle = np.append(front_edge, np.append(membrane_hist_edges_middle, back_edge))
            membrane_hist = np.append(0, np.append(self.membrane_hist, 0))
            return np.interp(x, membrane_hist_edges_middle, membrane_hist)
        def solvent_hist_func(x):
            solvent_hist_edges_middle = self.solvent_hist_edges[0:-1] + np.diff(self.solvent_hist_edges)/2
            front_edge = solvent_hist_edges_middle[0] * 2 - solvent_hist_edges_middle[1]
            back_edge = solvent_hist_edges_middle[-1] * 2 - solvent_hist_edges_middle[-2]
            solvent_hist_edges_middle = np.append(front_edge, np.append(solvent_hist_edges_middle, back_edge))
            solvent_hist = np.append(0, np.append(self.solvent_hist, 0))
            return np.interp(x, solvent_hist_edges_middle, solvent_hist)
        self.membrane_hist_func = membrane_hist_func
        self.solvent_hist_func = solvent_hist_func
        if self.verbose:
            print("Histogram functions defined after " + str(time.time() - self.start) + " seconds")




    def _readPositions(self, resnames):
        """
        Read the positions of atoms with specified residue names.

        Parameters:
        - resnames (list): List of residue names.

        Returns:
        - positions (numpy.ndarray): Array of atom positions.
        """
        atoms = self.Universe.select_atoms('resname ' + ' or resname '.join(resnames))
        positions = np.zeros((self.Universe.trajectory.n_frames,len(atoms),3))
        for ts in self.Universe.trajectory:
            positions[ts.frame,:,:] = atoms.positions
        return positions
    
    def analyseConstraints(self):
        """
        Analyze the constraints on the atom positions.

        Parameters:
        - y_min (float): Minimum value for the y-axis constraint.
        - y_max (float): Maximum value for the y-axis constraint.
        - z_min (float): Minimum value for the z-axis constraint.
        - z_max (float): Maximum value for the z-axis constraint.
        """
        plt.figure()
        plt.hist(self.membrane_atom_positions[:,:,0].flatten())
        plt.title('Histogram for x-axis')
        plt.xlabel('X-axis in Angstroms')
        plt.ylabel('Frequency')

        plt.figure()
        plt.hist(self.membrane_atom_positions[:,:,1].flatten(), bins=100)
        plt.legend()
        plt.axvline(x=self.y_middle - self.y_range/2, color='r', linestyle='--')  # y_min line
        plt.axvline(x=self.y_middle + self.y_range/2, color='r', linestyle='--')  # y_max line
        plt.title('Histogram for y-axis')
        plt.xlabel('X-axis in Angstroms')
        plt.ylabel('Frequency')

        plt.figure()
        plt.hist(self.membrane_atom_positions[:,:,2].flatten(), bins=100)
        plt.axvline(x=self.z_min, color='r', linestyle='--')  # z_min line
        plt.axvline(x=self.z_max, color='r', linestyle='--')  # z_max line
        plt.title('Histogram for z-axis')
        plt.xlabel('X-axis in Angstroms')
        plt.ylabel('Frequency')
        # plt.show()

    def _find_zConstraints(self, membrane_atom_positions):
        z_min = membrane_atom_positions[:,:,2].min()
        z_max = membrane_atom_positions[:,:,2].max()
        z_max = (z_min + z_max) / 2 + (z_max - z_min) / 2 * 0.8
        z_min = (z_min + z_max) / 2 - (z_max - z_min) / 2 * 0.8
        return z_min, z_max
    
    def _find_yConstraints(self, membrane_atom_positions):
        """y_profile = stats.gaussian_kde(membrane_atom_positions[:,:,1].flatten())
        plt.figure()
        x = np.linspace(membrane_atom_positions[:,:,1].min(), membrane_atom_positions[:,:,1].max(), 1000)
        plt.plot(x, y_profile(x))
        plt.plot()
        def holefunc(x):
            return x - y_profile(x) 
        y_middle, y_range = 0, 0
        return y_middle, y_range"""
        return self.y_middle, self.y_range
    
    def _filterPositions(self, positions, y_middle, y_range, z_min, z_max):
        """
        Filter positions based on y and z constraints.

        Parameters:
        positions (ndarray): Array of positions with shape (timesteps, atom_amount, 3).
        y_min (float): lower y constraint.
        y_max (float): upper y constraint.
        z_min (float): lower z constraint.
        z_max (float): upper z constraint.

        Returns:
        ndarray: Array of filtered x positions.
        """
        filtered_indices = np.where(
            (positions[:, :, 1] >= y_middle - y_range/2) & (positions[:, :, 1] <= y_middle+y_range/2) &
            (positions[:, :, 2] >= z_min) & (positions[:, :, 2] <= z_max)
        )
        filtered_x_positions = positions[filtered_indices[0], filtered_indices[1], 0].flatten()
        return filtered_x_positions


    def calculate_effective_pore_size(self, strategy='averaging'):
        """
        Calculate the effective pore size.

        Returns:
        - pore_size (float): The calculated effective pore size.
        """
        self.lower_edge, self.upper_edge = self._calculate_pore_edges(strategy=strategy)
        effective_pore_size = np.abs(self.lower_edge - self.upper_edge)  # in Angstroms
        return effective_pore_size
    
    def _calculate_pore_edges(self, strategy='averaging'):
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
        if strategy == 'averaging':
            first_zero_bin = np.where(self.membrane_hist == 0)[0][0]
            first_zero_middle = (self.membrane_hist_edges[first_zero_bin] + self.membrane_hist_edges[first_zero_bin + 1]) / 2
            first_non_zero_bin = np.where(self.solvent_hist > 0)[0][0]
            first_non_zero_middle = (self.solvent_hist_edges[first_non_zero_bin] + self.solvent_hist_edges[first_non_zero_bin + 1]) / 2
            avrg_lower_edge = np.abs(first_zero_middle + first_non_zero_middle) / 2

            last_zero_bin = np.where(self.membrane_hist == 0)[0][-1]
            last_zero_middle = (self.membrane_hist_edges[last_zero_bin] + self.membrane_hist_edges[last_zero_bin + 1]) / 2
            last_non_zero_bin = np.where(self.solvent_hist > 0)[0][-1]
            last_non_zero_middle = (self.solvent_hist_edges[last_non_zero_bin] + self.solvent_hist_edges[last_non_zero_bin + 1]) / 2
            avrg_upper_edge = np.abs(last_zero_middle + last_non_zero_middle) / 2

        
        elif strategy == 'intersection':
            """
            calculate pore size boarders by calculating the intersection of the two histograms
            """
            # Find the last intersection point of the histograms
            
            def intersection(x):
                return self.membrane_hist_func(x) - self.solvent_hist_func(x)
            
            min_x_membrane = self.membrane_hist_edges[0]
            max_x_membrane = self.membrane_hist_edges[-1]
            # print(min_x_membrane, max_x_membrane)
            avrg_lower_edge = optimize.root_scalar(intersection, bracket=[min_x_membrane, (max_x_membrane + min_x_membrane)/2], method='brentq').root
            avrg_upper_edge = optimize.root_scalar(intersection, bracket=[(max_x_membrane + min_x_membrane)/2, max_x_membrane ], method='brentq').root

        return avrg_lower_edge, avrg_upper_edge
    

    def plot(self):
        """
        Plot the data.

        Returns:
        - None
        """
        plt.figure()
        x = np.linspace(self.membrane_hist_edges[0], self.membrane_hist_edges[-1], 1000)
        plt.plot(x, self.membrane_hist_func(x), label='Membrane')
        plt.plot(x, self.solvent_hist_func(x), label='Solvent')
        plt.axvline(x=self.lower_edge, color='r', linestyle='--')
        plt.axvline(x=self.upper_edge, color='r', linestyle='--')
        plt.xlabel('X-axis in Angstroms')
        plt.ylabel('Frequency')
        plt.title('Histogram Line Plot along the X-axis with Y and Z constraints for Atoms')
        plt.grid(True)
        plt.legend()
        # plt.show()

    def analyseDensity(self):
        membrane_pos = self.membrane_atom_positions
        solvent_pos = self.solvent_atom_positions
        # combined_positions = np.concatenate((self.solvent_atom_positions, self.membrane_atom_positions), axis=1)

        filtered_indices_mem = np.where(
            (membrane_pos[:, :, 2] >= (self.z_min + self.z_max)/2 - 5) & 
            (membrane_pos[:, :, 2] <= (self.z_min + self.z_max)/2 + 5)
        )
        filtered_indices_sol = np.where(
            (solvent_pos[:, :, 2] >= (self.z_min + self.z_max)/2 - 5) & 
            (solvent_pos[:, :, 2] <= (self.z_min + self.z_max)/2 + 5)
        )

        filtered_positions_mem = membrane_pos[filtered_indices_mem[0], filtered_indices_mem[1], 0:2]
        filtered_positions_sol = solvent_pos[filtered_indices_sol[0], filtered_indices_sol[1], 0:2]


        kde_mem = gaussian_kde(filtered_positions_mem[::50,:].T)
        kde_sol = gaussian_kde(filtered_positions_sol[::50,:].T)

        # Evaluate the KDE on a grid
        xmin = min(filtered_positions_mem[:, 0])
        xmax = max(filtered_positions_mem[:, 0])
        xn = 100
        ymin = min(filtered_positions_mem[:, 1])
        ymax = max(filtered_positions_mem[:, 1])
        yn = 100
        x = np.linspace(xmin, xmax, xn)
        y = np.linspace(ymin, ymax, yn)
        X, Y = np.meshgrid(x, y)
        Z_mem = np.reshape(kde_mem([X.ravel(), Y.ravel()]), X.shape)
        Z_sol = np.reshape(kde_sol([X.ravel(), Y.ravel()]), X.shape)
        Z_sol_norm = Z_sol / np.max(Z_sol)
        Z_mem_norm = Z_mem / np.max(Z_mem)
        Z = Z_mem_norm + Z_sol_norm
        plt.figure()
        plt.pcolormesh(X, Y, Z, shading='gouraud')
        plt.colorbar()
        plt.axis('equal')