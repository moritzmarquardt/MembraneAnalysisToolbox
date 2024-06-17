import MDAnalysis as mda
import matplotlib.pyplot as plt
import os
import numpy as np
import MembraneAnalysisToolbox.funcs as tfm
import MembraneAnalysisToolbox.plot as tfmp
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.optimize import least_squares

class TransitionPathAnalysis:
    def __init__(
            self, 
            topology_file: str, 
            trajectory_file: str = None,
            verbose = False,
            ):
        
        self.topology_file = topology_file
        self.trajectory_file = trajectory_file
        self.verbose = verbose

        # Check if the files exist and have the correct file format
        if not os.path.exists(topology_file) or not topology_file.endswith('.tpr'):
            raise FileNotFoundError("Topology file does not exist or has wrong file format: A .tpr file is required.")
        if os.path.exists(trajectory_file) and not trajectory_file.endswith('.xtc'):
            raise FileNotFoundError("Trajectory file does not exist or has wrong file format: A .xtc file is required.")

        # Build Universe using the MDAnalysis library
        self.u = mda.Universe(topology_file, trajectory_file)
        # Define how many (nth) frames are being analysed
        # Here we calculate the nth based on the requirement that at least 5 frames are stored per ns of the simulation
        self.nth = int(np.floor(1000/self.u.trajectory.dt/5))
        # Based on the nth, we calculate the number of analysed timesteps
        self.timesteps = int(np.ceil(self.u.trajectory.n_frames/self.nth))
        if self.verbose:
            print("Universe: " + str(self.u))
            print("number of frames: " + str(self.u.trajectory.n_frames))
            print("step size: " + str(self.u.trajectory.dt) + " ps")
            print("simulation duration: " + str((self.u.trajectory.n_frames-1)*self.u.trajectory.dt/1000) + " ns")
            print("every " + str(self.nth) + "th frame is stored")
            print("number of analysed frames: " + str(self.timesteps))
            print("step size in analysis: " + str(self.u.trajectory.dt*self.nth) + " ps")

        
        # Trajectories of possible atoms/selectors will be stored in a dictionary
        # for speed and efficiency only trajectories of selectors that are actually 
        # needed will be loaded when accesing functions that need them
        self.trajectories = {}
        # Initialize timeline of the analysed frames
        # The timeline is mostly used for plotting purposes
        self.timeline = np.linspace(0, self.u.trajectory.n_frames*self.u.trajectory.dt, self.timesteps)




    def _allocateTrajectory(self, selector):
        """
        Save the trajectories of the selectors to the dictionary.

        Parameters:
        selector (str): The selector used to select atoms from the universe.

        Returns:
        None

        This method allocates and saves the trajectories of the selected atoms to the dictionary.
        It initializes an array to store the positions of the atoms over time and populates it
        by iterating over the trajectory frames. The resulting positions array is then stored
        in the `trajectories` dictionary under the given selector key.
        """
        atoms = self.u.select_atoms(selector)
        if self.verbose:
            print(selector + " loading...")
            print("number of " + selector + " atoms: " + str(atoms.n_atoms))
        positions = np.zeros((atoms.n_atoms, self.timesteps, 3))
        for i, ts in enumerate(self.u.trajectory[::self.nth]):
            positions[:,i,:] = atoms.positions
        self.trajectories[selector] = positions
        if self.verbose:
            print(selector + " loaded.")
    
    def inspect(self, selectors, z_lower=None, L=None):
        """
        Visualizes the histograms of the z, x, and y coordinates for the given selectors.

        Parameters:
            selectors (list or str): The selectors to inspect. If a single selector is provided as a string, it will be converted to a list.
            z_lower (float, optional): The lower bound of the z-coordinate range to display on the histogram. Defaults to None.
            L (float, optional): The length of the z-coordinate range to display on the histogram. Defaults to None.

        Returns:
            None
        """
        if type(selectors) is not list:
            selectors = [selectors]

        seles_not_loaded = [sele for sele in selectors if sele not in self.trajectories]
        if len(seles_not_loaded) > 0:
            for sele in seles_not_loaded:
                self._allocateTrajectory(sele)

        total_elements = sum(self.trajectories[sele].shape[0] * self.trajectories[sele].shape[1] for sele in selectors)
        x = np.empty(total_elements)
        y = np.empty(total_elements)
        z = np.empty(total_elements)
        index = 0
        # Collect data for each selector and store directly in the pre-allocated arrays
        for sele in selectors:
            positions = self.trajectories[sele]
            n_elements = positions.shape[0] * positions.shape[1]
            x[index:index + n_elements] = positions[:, :, 0].flatten()
            y[index:index + n_elements] = positions[:, :, 1].flatten()
            z[index:index + n_elements] = positions[:, :, 2].flatten()
            index += n_elements

        fig_z_dist, ax_z_dist = plt.subplots()
        fig_z_dist.suptitle("Histogram of z", fontsize="x-large")
        ax_z_dist.hist(z, bins=100, density=True, alpha=0.5, label=selectors)
        ax_z_dist.set_xlabel("z", fontsize="x-large")
        ax_z_dist.set_ylabel("Frequency", fontsize="x-large")
        if z_lower is not None and L is not None:
            ax_z_dist.axvline(z_lower, color='r', linestyle='--', label='z_lower')
            ax_z_dist.axvline(z_lower+L, color='r', linestyle='--', label='z_upper')
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

    def find_z_lower_hexstruc(self, mem_selector, L):
        """
        Find the z-coordinate of the lower boundary of the hexagonal structure
        
        Parameters:
            mem_selector (str or list): The selector for the membrane trajectory.
                If a list is provided, the first element will be used.
            L (float): The length of the hexagonal structure.
        
        Returns:
            float: The z-coordinate of the lower boundary of the hexagonal structure.
        """
        if type(mem_selector) is list:
            mem_selector = mem_selector[0]
        
        if mem_selector not in self.trajectories:
            self._allocateTrajectory(mem_selector)

        z = self.trajectories[mem_selector][:, :, 2].flatten()
        
        _, bins = np.histogram(z, bins=100, density=True)

        z_lower = bins[0]
        z_upper = bins[-1]
        z_middle = (z_lower + z_upper) / 2

        return z_middle - L/2


    def calc_passagetimes(self, selectors, z_lower, L):
        """
        Calculates the passage times for the given selectors.

        Args:
            selectors (list or str): The selectors to calculate passage times for.
            z_lower (float): The lower bound of the z-coordinate range.
            L (float): The length of the z-coordinate range.

        Returns:
            tuple: A tuple containing three arrays - concatenated first-first passage times (ffs),
                   concatenated first-final passage times (ffe), and concatenated indices (indizes).
        """
        if type(selectors) is not list:
            selectors = [selectors]

        seles_not_loaded = [sele for sele in selectors if sele not in self.trajectories]
        if len(seles_not_loaded) > 0:
            for sele in seles_not_loaded:
                print(sele + " loading...")
                self._allocateTrajectory(sele)

        ffe = []
        ffs = []
        indizes = []
        for sele in selectors:
            z = self.trajectories[sele][:,:,2]
            ffs_sele, ffe_sele, indizes_sele = tfm.dur_dist_improved(z, [z_lower, z_lower+L])
            ffs.append(ffs_sele)
            ffe.append(ffe_sele)
            indizes.append(indizes_sele)
        
        return np.concatenate(ffs), np.concatenate(ffe), np.concatenate(indizes)
    

    def plot_passagetimedist(self, passage_times):
        plt.figure("Verteilung der Durchgangszeiten")
        tfmp.plot_dist(passage_times,number_of_bins=10, max_range=np.max(passage_times))
        plt.xlabel("Durchgangszeiten")
        plt.ylabel("relative Häufigkeit")


    def calc_diffusion(self, passage_times, L, T):
        """
        Calculate the diffusion coefficient using the methods of Gotthold Fläschner.

        Args:
            passage_times (array-like): Array of passage times.
            L (float): Length parameter.
            T (float): Temperature parameter.

        Returns:
            float: The diffusion coefficient.

        """
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

        # plot PDF

        """ PREPARE DATA """
        bins = int(10*np.max(passage_times)/centertime)
        histo, edges = np.histogram(passage_times, bins, density=True);
        center=edges-(edges[2]-edges[1]);
        center=np.delete(center,0)
        edges=np.delete(edges,0)

        """ PLOT DATA """
        x = x_cdf
        y2 = self.fitfunc_hom(x,params_hom_cdf[0], L)

        ax1.hist(passage_times,bins=len(center), density=True, color=[0,0.5,0.5])
        ax1.plot(x[1:],y2[1:],label='hom', color='red', ls='dashed')
        ax1.set_xlim(0,x_lim)
        ax1.set_ylim(0,1.2*np.max(histo[0:]))
        ax1.legend(loc='center right')

        return D_hom_cdf
    
#########################################################################################
# Funktionen aus Gottholds Skript #######################################################
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
    
    def fitfunc_hom(self,x,D,L):
        i=151
        result=0
        for j in range(1,i):
            result=result+self.hom(x,D,j,L)
        return(2*np.pi**2*D/(L)**2*result)
    
    def hom(self,x,D,i,L):
        t=(L)**2/(i**2*np.pi**2*D)
        return((-1)**(i-1)*i**2*np.exp(-x/t))
# Ende Funktionen aus Gottholds Skript ##################################################
#########################################################################################


