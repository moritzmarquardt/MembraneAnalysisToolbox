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
        Safe the trajectories of the selectors to the dictionary
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
    
    def inspect(self, selectors):
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

    def find_z_limits(self, mem_selector):
        if type(mem_selector) is not list:
            mem_selector = [mem_selector]

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
        ax_z_dist.legend()


    def calc_passagetimes(self, selectors, z_lower, L):
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
        # Calculate the diffusion coefficient using the mehtods of Gotthold Fläschner
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
# Ende Funktionen aus Gottholds Skript ##################################################
#########################################################################################


