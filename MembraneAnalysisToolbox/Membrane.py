from abc import ABC, abstractmethod
import os
import numpy as np
import MDAnalysis as mda

##unvollständig, idee: membran as klasse, idee ist in einer onenote von scicomp abgelegt
# idee: ersetzen von scripten step1 -4 durch eine klasse, die die methoden enthält

class MembraneSimulation(ABC):
    def __init__(
            self,
            topology_path: str,
            trajectory_path: str,
            output_path: str,
            membrane_resnames = None,
            solvent_resnames = None
            ):
        if not os.path.exists(topology_path) or not os.path.exists(trajectory_path):
            raise Exception("topol.tpr or traj.xtc file not found in the specified path -> storing is not possible")
        
        self.u = mda.Universe(topology_path, trajectory_path)

        self.params = {
            "topology_path": topology_path,
            "trajectory_path": trajectory_path,
            "output_path": output_path,
            "number_of_frames": self.u.trajectory.n_frames,
            "step_size": self.u.trajectory.dt,
            "simulation_duration": (self.u.trajectory.n_frames-1)*self.u.trajectory.dt/1000,
            "number_of_hex_atoms": hex.n_atoms,
            "number_of_dod_atoms": dod.n_atoms,
            "nth": nth,
        }

        pass

    @abstractmethod
    def method1(self):
        # Method 1 code here
        pass
    
    def method2(self):
        # Method 2 code here
        pass
    
    # Add more methods as needed