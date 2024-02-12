import MDAnalysis as mda
import matplotlib.pyplot as plt
import os
import numpy as np

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
            z_min = None, 
            z_max = None
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
        
        if os.path.splitext(topology_file)[1] == '.tpr' and trajectory_file is None:
            raise ValueError("A trajectory file is required when using a .tpr topology file.")
        if not (os.path.splitext(topology_file)[1] == '.tpr' or os.path.splitext(topology_file)[1] == '.gro'):
            raise ValueError("A topology file is required. Either a gro or a tpr file.")
        self.membrane_resnames = membrane_resnames
        self.solvent_resnames = solvent_resnames
        
        self.Universe = mda.Universe(topology_file, trajectory_file)

        self.membrane_atom_positions = self._readPositions(self,membrane_resnames)
        self.solvent_atom_positions = self._readPositions(self,solvent_resnames)

        self.membrane_atom_positions_filtered = self._filterPositions(self.membrane_atom_positions, y_middle, y_range, z_min, z_max)
        self.solvent_atom_positions_filtered = self._filterPositions(self.solvent_atom_positions, y_middle, y_range, z_min, z_max)

        self.bins = 100
        self.membrane_hist, self.membrane_hist_edges = np.histogram(self.membrane_atom_positions_filtered, density=1,bins=self.bins)
        self.solvent_hist, self.solvent_hist_edges = np.histogram(self.solvent_atom_positions_filtered, density=1,bins=self.bins)
        




    def _readPositions(self, resnames):
        """
        Read the positions of atoms with specified residue names.

        Parameters:
        - resnames (list): List of residue names.

        Returns:
        - positions (numpy.ndarray): Array of atom positions.
        """
        atoms = self.Universe.select_atoms('resname ' + ' or resname '.join(resnames))
        positions = np.zeros((self.Universe.trajectory.n_frames),len(atoms),3)
        for ts in self.Universe.trajectory:
            positions[ts.frame,:,:] = atoms.positions
        return positions
    
    def analyseConstraints(self,y_middle, y_range, z_min, z_max):
        """
        Analyze the constraints on the atom positions.

        Parameters:
        - y_min (float): Minimum value for the y-axis constraint.
        - y_max (float): Maximum value for the y-axis constraint.
        - z_min (float): Minimum value for the z-axis constraint.
        - z_max (float): Maximum value for the z-axis constraint.
        """
        plt.figure()
        plt.hist(self.membrane_atom_positions[:,:,0].flatten(),bins=50)
        plt.title('Histogram for x-axis')
        plt.figure()
        plt.hist(self.membrane_atom_positions[:,:,1].flatten(),bins=50)
        plt.axvline(x=y_middle - y_range/2, color='r', linestyle='--')  # y_min line
        plt.axvline(x=y_middle + y_range/2, color='r', linestyle='--')  # y_max line
        plt.title('Histogram for y-axis')
        plt.figure()
        plt.hist(self.membrane_atom_positions[:,:,2].flatten(),bins=50)
        plt.axvline(x=z_min, color='r', linestyle='--')  # z_min line
        plt.axvline(x=z_max, color='r', linestyle='--')  # z_max line
        plt.title('Histogram for z-axis')
        plt.show()

    def findConstraints(self):
        """
        Find the best constraints for filtering positions.

        Returns:
        - constraints (tuple): Tuple of best constraints (y_min, y_max, z_min, z_max).
        """
        # TODO: create method to automatically find best constraints
        return None
    
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


    def calculate_effective_pore_size(self):
        """
        Calculate the effective pore size.

        Returns:
        - pore_size (float): The calculated effective pore size.
        """
        self.lower_edge, self.upper_edge = self._calculate_pore_size()
        effective_pore_size = np.abs(self.lower_edge - self.upper_edge)  # in Angstroms
        return effective_pore_size
    
    def _calculate_pore_size(self):
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

        return avrg_lower_edge, avrg_upper_edge

    def plot(self):
        """
        Plot the data.

        Returns:
        - None
        """
        plt.figure()
        plt.plot(self.membrane_hist_edges[:-1], self.membrane_hist, label=' & '.join(self.membrane_resnames), linestyle='-', marker='o', markersize=3)
        plt.plot(self.solvent_hist_edges[:-1], self.solvent_hist, label=' & '.join(self.solvent_resnames), linestyle='-', marker='o', markersize=3)
        x1 = self.lower_edge
        x2 = self.upper_edge
        plt.axvline(x=x1, color='r', linestyle='--')
        plt.axvline(x=x2, color='r', linestyle='--')
        plt.xlabel('X-axis in Angstroms')
        plt.ylabel('Frequency')
        plt.title('Histogram Line Plot along the X-axis with Y and Z constraints for Atoms')
        plt.grid(True)
        plt.legend()
        plt.show()