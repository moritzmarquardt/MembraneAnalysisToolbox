import MDAnalysis as mda
import matplotlib.pyplot as plt
import os
import numpy as np
import MembraneAnalysisToolbox.funcs as tfm
import MembraneAnalysisToolbox.plot as tfmp
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.optimize import least_squares

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
            membrane_selectors = None,
            solvent_selectors = None,
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

        self.u = mda.Universe(topology_file, trajectory_file)
        #we want at least 5 steps per ns because transitions can last only 1ns and it needs steps to identify it as a passage
        self.nth = int(np.floor(1000/self.u.trajectory.dt/5))
        self.timesteps = int(np.ceil(self.u.trajectory.n_frames/self.nth))
        if self.verbose:
            print("Universe: " + str(self.u))
            print("number of frames: " + str(self.u.trajectory.n_frames))
            print("step size: " + str(self.u.trajectory.dt) + " ps")
            print("simulation duration: " + str((self.u.trajectory.n_frames-1)*self.u.trajectory.dt/1000) + " ns")
            print("every " + str(self.nth) + "th frame is stored")

        
        #STEP 1: Read the positions of the membrane and solvent atoms
        self.positions_dict = {}
        self.timeline = np.zeros(self.timesteps)
        # for i, mem_sel in enumerate(membrane_selectors):
        #     self.positions_dict[mem_sel] = self._readPositions(mem_sel, self.nth, self.timesteps)
        # for i, solv_sel in enumerate(solvent_selectors):
        #     self.positions_dict[solv_sel] = self._readPositions(solv_sel, self.nth, self.timesteps)
        # if self.verbose:
        #     print("Positions read")
        #     for selector, positions in self.positions_dict.items():
        #         print(f"Shape of positions for '{selector}': {positions.shape}")




    def _readPositions(self, selector, nth = 1, timesteps = None):
        """
        Read the positions of atoms with specified residue names.

        Parameters:
        - resname

        Returns:
        - positions (numpy.ndarray): Array of atom positions.
        """
        atoms = self.u.select_atoms('resname ' + selector)
        if self.verbose:
            print("number of " + selector + " atoms: " + str(atoms.n_atoms))
        positions = np.zeros((atoms.n_atoms, timesteps, 3))
        if self.timeline[1] == 0: # timeline has not been initialized yet
            for i, ts in enumerate(self.u.trajectory[::nth]):
                positions[:,i,:] = atoms.positions
                self.timeline[i] = ts.time
        else: # timeline has been initialized
            for i, ts in enumerate(self.u.trajectory[::nth]):
                positions[:,i,:] = atoms.positions
            
        return positions
    
    def inspect(self, selectors):
        seles_not_loaded = [sele for sele in selectors if sele not in self.positions_dict]
        if len(seles_not_loaded) > 0:
            for sele in seles_not_loaded:
                print(sele + " loading...")
                self.positions_dict[sele] = self._readPositions(sele, self.nth, self.timesteps)

        x = []
        y = []
        z = []
        for sele in selectors:
            x.extend(self.positions_dict[sele][:,:,0].flatten())
            y.extend(self.positions_dict[sele][:,:,1].flatten())
            z.extend(self.positions_dict[sele][:,:,2].flatten())
        
        # print(z)

        fig_z_dist, ax_z_dist = plt.subplots()
        fig_z_dist.suptitle("Histogram of z", fontsize="x-large")
        ax_z_dist.hist(z, bins=100, density=True, alpha=0.5, label=selectors)
        ax_z_dist.set_xlabel("z", fontsize="x-large")
        ax_z_dist.set_ylabel("Frequency", fontsize="x-large")
        ax_z_dist.legend()

        fig_x_dist, ax_x_dist = plt.subplots()
        fig_x_dist.suptitle("Histogram of x", fontsize="x-large")
        ax_x_dist.hist(x, bins=100, density=True, alpha=0.5, label=selectors)
        ax_x_dist.set_xlabel("x", fontsize="x-large")
        ax_x_dist.set_ylabel("Frequency", fontsize="x-large")
        ax_x_dist.legend()

        fig_y_dist, ax_y_dist = plt.subplots()
        fig_y_dist.suptitle("Histogram of y", fontsize="x-large")
        ax_y_dist.hist(y, bins=100, density=True, alpha=0.5, label=selectors)
        ax_y_dist.set_xlabel("y", fontsize="x-large")
        ax_y_dist.set_ylabel("Frequency", fontsize="x-large")
        ax_y_dist.legend()

        plt.show()


    def calc_passagetimes(self, selectors, z_lower, z_upper):
        seles_not_loaded = [sele for sele in selectors if sele not in self.positions_dict]
        if len(seles_not_loaded) > 0:
            for sele in seles_not_loaded:
                print(sele + " loading...")
                self.positions_dict[sele] = self._readPositions(sele, self.nth, self.timesteps)

        ffe = []
        ffs = []
        indizes = []
        for sele in selectors:
            z = self.positions_dict[sele][:,:,2]
            ffs_sele, ffe_sele, indizes_sele = tfm.dur_dist_improved(z, [z_lower, z_upper])
            ffs.append(ffs_sele)
            ffe.append(ffe_sele)
            indizes.append(indizes_sele)
        
        return ffs, ffe, indizes
    
    def calc_diffusion(self, passage_times, L, T):
        ecdf = ECDF(passage_times)
        idx = (np.abs(ecdf.y - 0.5)).argmin()
        centertime = ecdf.x[idx]

        """ FIT DATA """
        params_hom_cdf = self.fitting_hom_cdf_lsq(ecdf.x[1:],ecdf.y[1:], L)

        """ PLOT DATA """
        x_lim = centertime*4
        x_cdf = np.linspace(0,x_lim*2,200)
        y_hom_cdf = self.fitfunc_hom_cdf(x_cdf, params_hom_cdf[0], L)

        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('PDF and CDF fit')
        ax2.scatter(ecdf.x, ecdf.y, color=[0,0.5,0.5])
        ax2.plot(x_cdf[1:],y_hom_cdf[1:],label='hom', color='red', ls = 'dashed')
        ax2.legend(loc='center right')
        ax2.set_xlim(0,x_lim)

        D_hom_cdf=params_hom_cdf[0]
        return D_hom_cdf
    
    def plot_passagetimedist(self, ffs, ffe, indizes):
        pass


    # Funktionen aus Gottholds Skript #####################################################
    def hom_cdf(self,x,D,i,L):
        t=(L)**2/(i**2*np.pi**2*D) #L^2/(i^2*pi^2*D)
        return((-1)**(i-1)*np.exp(-x/t)) #summand in Gl. 10 vanHijkoop

    def fitfunc_hom_cdf(self,x,D,L):
        i=50 #Summe geht bis 50 (approx statt undendlich)
        result=0
        for j in range(1,i):
            result=result+self.hom_cdf(x,D,j,L)
        return(1-2*result) #gleichung 10 in vanHijkoop paper

    def fitfunc_hom_cdf_lsq(self,L):
        def f(D,x,y):
            i=50
            result=0
            for j in range(1,i):
                result=result+self.hom_cdf(x,D,j,L)
            return(1-2*result-y)
        return f

    def fitting_hom_cdf_lsq(self,x_data,y_data, L):
        res_robust = least_squares(self.fitfunc_hom_cdf_lsq(L), x0=20, loss ='soft_l1', f_scale=0.3, args=(x_data, y_data))
        return res_robust.x
    # Ende Funktionen aus Gottholds Skript ##################################################


    def plot_passagetimedist(self, passage_times):
        plt.figure("Verteilung der Durchgangszeiten")
        tfmp.plot_dist(passage_times,number_of_bins=10, max_range=np.max(passage_times))
        plt.xlabel("Durchgangszeiten")
        plt.ylabel("relative Häufigkeit")