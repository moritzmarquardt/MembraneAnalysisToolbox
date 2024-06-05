import MDAnalysis as mda
import matplotlib.pyplot as plt
import os
import numpy as np

class TransitionPathAnalysis:
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
            solvent_resname = None,
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
        
        if os.path.splitext(topology_file)[1] == '.tpr' and trajectory_file is None:
            raise ValueError("A trajectory file is required when using a .tpr topology file.")
        if not (os.path.splitext(topology_file)[1] == '.tpr' or os.path.splitext(topology_file)[1] == '.gro'):
            raise ValueError("A topology file is required. Either a gro or a tpr file.")
        self.verbose = verbose

        self.Universe = mda.Universe(topology_file, trajectory_file)
        if self.verbose:
            print("Universe: " + str(self.Universe))
            print("timesteps: " + str(self.Universe.trajectory.n_frames))

        self.membrane_atom_positions = self._readPositions(membrane_resnames)
        self.solvent_atom_positions = self._readPositions(solvent_resname)
        if self.verbose:
            print("Positions read")




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
    
    def analyseStructure(self):
        """
        Analyze the constraints on the atom positions.

        Parameters:
        - y_min (float): Minimum value for the y-axis constraint.
        - y_max (float): Maximum value for the y-axis constraint.
        - z_min (float): Minimum value for the z-axis constraint.
        - z_max (float): Maximum value for the z-axis constraint.
        """
        plt.figure()
        plt.hist(self.membrane_atom_positions[:,:,0].flatten(), bins=100)
        plt.title('Histogram for x-axis')
        plt.xlabel('X-axis in Angstroms')
        plt.ylabel('Frequency')

        plt.figure()
        plt.hist(self.membrane_atom_positions[:,:,1].flatten(), bins=100)
        plt.legend()
        plt.title('Histogram for y-axis')
        plt.xlabel('X-axis in Angstroms')
        plt.ylabel('Frequency')

        plt.figure()
        plt.hist(self.membrane_atom_positions[:,:,2].flatten(), bins=100)
        plt.title('Histogram for z-axis')
        plt.xlabel('X-axis in Angstroms')
        plt.ylabel('Frequency')
        plt.show()


class cubeStructure:
    def __init__(
            self,
            Lx: float,
            Ly: float,
            Lz: float,
            Nx: int,
            Ny: int,
            Nz: int,
            ):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz