import MDAnalysis as mda
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import MembraneAnalysisToolbox.funcs as tfm
import MembraneAnalysisToolbox.plot as tfmp
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.optimize import least_squares

class TransitionPathAnalysis:
    """
    A class for performing analysis on molecular dynamics simulation trajectories.

    Parameters:
        topology_file (str): The path to the topology file (.tpr) of the system.
        trajectory_file (str): The path to the trajectory file (.xtc) of the system. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Attributes:
        topology_file (str): The path to the topology file (.tpr) of the system.
        trajectory_file (str): The path to the trajectory file (.xtc) of the system.
        verbose (bool): Whether to print verbose output.
        u (mda.Universe): The MDAnalysis Universe object representing the system.
        nth (int): The number of frames to analyze per timestep in order to get 5 frames per nanosecond.
        timesteps (int): The number of analyzed timesteps.
        trajectories (dict): A dictionary to store the trajectories of selectors (e.g. atoms of molecules or beads). A selector is used to select atoms from the universe using the MDA lingo.
        timeline (np.ndarray): An array representing the timeline of the analyzed frames. Used for plotting purposes.

    """
    def __init__(
            self, 
            topology_file: str, 
            trajectory_file: str,
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
        # Print some information about the simulation
        if self.verbose:
            print("Information about the simulation:")
            print("\t- number of atoms: " + str(self.u.atoms.n_atoms))
            print('\t- number of frames: ' + str(self.u.trajectory.n_frames))
            print("\t- step size: " + str(self.u.trajectory.dt) + " ps")
            print("\t- simulation duration: " + str((self.u.trajectory.n_frames-1)*self.u.trajectory.dt/1000) + " ns")
            print("Information about the analysis:")
            print("\t- every " + str(self.nth) + "th frame is stored")
            print("\t- number of analysed frames: " + str(self.timesteps))
            print("\t- step size in analysis: " + str(self.u.trajectory.dt*self.nth) + " ps\n")

        
        # Trajectories of possible atoms/selectors will be stored in a dictionary
        # for speed and efficiency only trajectories of selectors that are actually 
        # needed will be loaded when accesing functions that need them
        self.trajectories = {}
        # Initialize timeline of the analysed frames
        # The timeline is mostly used for plotting purposes
        self.timeline = np.linspace(0, self.u.trajectory.n_frames*self.u.trajectory.dt, self.timesteps)

    def _allocateTrajectories(self, selectors):
        """
        Allocates the trajectories of the given selectors in the trajectories dictionary.

        Parameters:
            selectors (list or str): The selectors to allocate trajectories for. If a single selector is provided as a string, it will be converted to a list.

        Raises:
            ValueError: If the selectors parameter is not a string or a list of strings.
        """
        # check for the format of the selectors
        if isinstance(selectors, str):
            selectors = [selectors]
        elif not isinstance(selectors, list):
            raise ValueError("Selectors must be a string or a list of strings.")
        
        selectors_unstored = [selector for selector in selectors if selector not in self.trajectories]
        if self.verbose:
            print("Allocating trajectories for selectors: \"" + "\", \"".join(selectors_unstored) + "\".")
        atomslist = [self.u.select_atoms(selector) for selector in selectors_unstored]
        positions =  np.zeros((sum([atoms.n_atoms for atoms in atomslist]), self.timesteps, 3))
        indexes = [0]
        for atoms in atomslist:
            indexes.append(indexes[-1] + atoms.n_atoms)
        for i, _ in enumerate(self.u.trajectory[::self.nth]):
            if self.verbose:
                percentage = int((i+1)/self.timesteps*100)
                sys.stdout.write(f"\r\tProgress: {percentage}%")
                sys.stdout.flush()
            for j, atoms in enumerate(atomslist):
                positions[indexes[j]:indexes[j+1],i,:] = atoms.positions
        for i, sele in enumerate(selectors_unstored):
            self.trajectories[sele] = positions[indexes[i]:indexes[i+1],:,:]

        if self.verbose:
            print("\nTrajectories allocated.")

    
    def inspect(self, selectors, z_lower=None, L=None):
        """
        Visualizes the histograms of the z, x, and y coordinates for the given selectors.

        Parameters:
            selectors (list or str): The selectors to inspect. If a single selector is provided as a string, it will be converted to a list.
            z_lower (float, optional): The lower bound of the z-coordinate range to display on the histogram. Defaults to None.
            L (float, optional): The length of the z-coordinate range to display on the histogram. Defaults to None.

        Returns:
            tuple: A tuple containing the figures for the histograms of x, y and z coordinates.

        Raises:
            ValueError: If the selectors parameter is not a string or a list of strings.

        """
        self._allocateTrajectories(selectors)

        # calculate the total number of elements in the trajectories of the selectors to be able to allocate flat arrays for the histograms
        total_elements = sum(self.trajectories[sele].shape[0] * self.trajectories[sele].shape[1] for sele in selectors)
        x = np.empty(total_elements)
        y = np.empty(total_elements)
        z = np.empty(total_elements)
        # Collect data for each selector and store directly in the pre-allocated arrays
        index = 0
        for sele in selectors:
            positions = self.trajectories[sele]
            n_elements = positions.shape[0] * positions.shape[1]
            x[index:index + n_elements] = positions[:, :, 0].flatten()
            y[index:index + n_elements] = positions[:, :, 1].flatten()
            z[index:index + n_elements] = positions[:, :, 2].flatten()
            index += n_elements

        # Create histograms for the x, y, and z coordinates
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

    def find_z_lower_hexstruc(self, mem_selector: str, L):
        """
        Find the z-coordinate of the lower boundary of the hexagonal structure.
        This is done by calculating a histogram of the z-coordinates of the membrane selectors.
        Then the max and min value of the histogram are used to calculate the middle value.
        Then the lower boundary is calculated by subtracting half of the length of the hexagonal structure.
        
        Parameters:
            mem_selector (str): The selector for the membrane trajectory.
                If a list is provided, the first element will be used.
            L (float): The length of the hexagonal structure.
        
        Returns:
            float: The z-coordinate of the lower boundary of the hexagonal structure.
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
        self._allocateTrajectories(selectors)

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


    def calc_diffusion(self, passage_times, L, plot=True):
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

        D_hom_cdf=params_hom_cdf[0]

        if plot:

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
    
    def bootstrap_diffusion(self, selector, z_lower, L, n_bootstraps, plot=True):
        #first do the bootstrapping for only one element
        if not isinstance(selector, str):
            raise ValueError("Selector must be a string.")
        
        self._allocateTrajectories(selector)
        bootstrap_pieces = np.array_split(self.trajectories[selector][:,:,2], n_bootstraps, axis=0) #idk if this works
        # print(bootstrap_pieces)
        # print(bootstrap_pieces[0].shape)
        bootstrap_diffusions = np.zeros(n_bootstraps)
        for i, piece in enumerate(bootstrap_pieces):
            ffs, ffe, indizes = tfm.dur_dist_improved(piece, [z_lower, z_lower+L])
            bootstrap_diffusions[i] = self.calc_diffusion(ffe-ffs, L, plot=plot)
        return bootstrap_diffusions
    
    def bootstrapping_diffusion(self, selector, bootstrap_sample_length_ns, n_bootstraps, z_lower, L, plot=True):
        if not isinstance(selector, str):
            raise ValueError("Selector must be a string.")
        
        self._allocateTrajectories(selector)

        bootstrap_sample_length = bootstrap_sample_length_ns *1000 #convert to ps
        bootstrap_sample_length /= self.u.trajectory.dt #convert to frames
        bootstrap_sample_length /= self.nth #convert to steps in the analysis
        bootstrap_sample_length = int(bootstrap_sample_length) #convert to int

        bootstrap_diffusions = np.zeros(n_bootstraps)
        rg = np.random.default_rng() #this is numpys new state of the art random number generator routine according to their docs
        if self.verbose:
            print(f"Bootstrapping for {n_bootstraps} samples of length {bootstrap_sample_length} steps.")
        for i in range(n_bootstraps):
            sample_start_index = rg.integers(0, self.timesteps - bootstrap_sample_length)
            # print(i, sample_start_index)
            progress = int(i/n_bootstraps*100)
            if self.verbose:
                sys.stdout.write(f"\r\tProgress: {progress}%")
                sys.stdout.flush()
            sample = self.trajectories[selector][:, sample_start_index:sample_start_index+bootstrap_sample_length, 2]
            ffs, ffe, _ = tfm.dur_dist_improved(sample, [z_lower, z_lower+L])
            bootstrap_diffusions[i] = self.calc_diffusion(ffe-ffs, L, plot=plot)
            
        if self.verbose:
            print("\nBootstrapping finished.")

        return bootstrap_diffusions

    
    

        
    
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


