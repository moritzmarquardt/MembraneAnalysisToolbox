import numpy as np
from scipy.optimize import basinhopping, curve_fit, dual_annealing, least_squares
from statsmodels.distributions.empirical_distribution import ECDF

"""
This is a collection of core functions that are used by the analysis classes,
but can also be used standalone as imported functions
"""


def calc_hor_dist(x_traj, y_traj, ffs, ffe) -> np.ndarray:
    """
    Calculate the horizontal x-y-distance that the molecule goes
    when going through the membrane

    Args:
        x_traj (_type_): all x trajectories of the passages
        y_traj (_type_): all y trajectories of the passages
        ffs (_type_): start times of the transitions
        ffe (_type_): end times of the transitions

    Raises:
        Exception: raises an exception if a pbc (periodic boundary condition) jump is detected

    Returns:
        _type_: horizontal distances of the passages
    """
    """
    x_traj and y_traj are all the x and y trajectories of the passages
    ffs and ffe are the start and endimes of the transitions
    """
    # TODO avoid using for loops and instead use numpy methods
    distances = []
    for i in range(x_traj.shape[0]):
        X = x_traj[i, :]
        Y = y_traj[i, :]
        distance = 0
        for t in np.arange(ffs[i], ffe[i]):
            # goes from start time til one step before end time.
            # this fits with the t+1 indexing below
            dist = np.sqrt(pow(X[t + 1] - X[t], 2) + pow(Y[t + 1] - Y[t], 2))
            if dist < 9:
                distance = distance + dist
            else:  # pbc jump handling
                raise Exception(
                    "pbc jump detected -- distance measurement is not possible"
                )
                dist_pbc = min(
                    np.sqrt(pow(X[t] - 18 - X[t - 1], 2) + pow(Y[t] - Y[t - 1], 2)),
                    np.sqrt(pow(X[t] + 18 - X[t - 1], 2) + pow(Y[t] - Y[t - 1], 2)),
                    np.sqrt(pow(X[t] - X[t - 1], 2) + pow(Y[t] - 18 - Y[t - 1], 2)),
                    np.sqrt(pow(X[t] - X[t - 1], 2) + pow(Y[t] + 18 - Y[t - 1], 2)),
                )  # approach of calculating the waled distance when the molecule jumped
                distance = distance + dist_pbc

        distances = np.append(distances, distance)

    return distances


def path_cat(X, Y, ffs, ffe):
    """
    find direct transitions

    Args:
        X (_type_): all x trajectories of the transitions
        Y (_type_): all y trajectories of the transitions
        ffs (_type_): start times of the transitions
        ffe (_type_): end times of the transitions

    Returns:
        _type_: indizes of direct transitions
    """
    d = []
    # cutoff distance which defines when a particle has left its direct path
    # TODO: avaoid hard coding 4.5
    # TODO: avoid using for loops and instead use numpy methods
    k = 4.5
    for i in range(X[:, 0].size):
        x0 = X[i, ffs[i]]
        y0 = Y[i, ffs[i]]
        indic = 0
        for t in np.arange(ffs[i], ffe[i]):
            dist = np.sqrt(pow(X[i, t + 1] - x0, 2) + pow(Y[i, t + 1] - y0, 2))
            if dist > k:
                indic = 1
        if indic != 1:
            d = np.append(d, i)

    return d.astype(int)


def findPassagesHexOptimised(T, lowerBound, upperBound) -> tuple:
    """
    find passage times, optimised and vectorised for hexagonal membranes
    optimised in the case that p=1 and p_middle=1
    """
    raise NotImplementedError("This function is not fully implemented and tested yet")
    # TODO: implement and test this function
    z_Trajs = T[:, :, 2]
    inside = (z_Trajs > lowerBound) & (z_Trajs < upperBound)
    diff = np.diff(inside, axis=1)
    ffs = np.where(diff == 1) - 1
    ffe = np.where(diff == -1) + 1
    ffi = np.zeros(ffs[0].size)
    return ffe, ffe, ffi


def findPassages(T, isAtomAbove, isAtomBelow, p=1, p_middle=1) -> tuple:
    """
    Measure start and endpoint of passages through the bounds.
    Returns the array of starting times and the array of endtimes (not in ns, but in timesteps!)

    Args:
        T (np.ndarray): trajectories: 3D array with shape (number of trajectories, number of timesteps, 3)
        isAtomAbove (function): function that returns True if the atom is above the membrane
        isAtomBelow (function): function that returns True if the atom is below the membrane
        p (int, optional): timesteps, that an object has to be above or below a
        bound to be seen as above or below. different values than 1 can make sense
        to compensate uncontinuous behavior (see documentation for more details). Defaults to 1.
        p_middle (int, optional): timesteps, that an object has to be in the middle to
        be seen as passaging through the middel. 3 means, the object has to be in
        the middle for 3 timesteps to be counted as a valid transition. Defaults to 1 because it is the
        most basic definition of a transition.

    Returns:
        _type_: flip start und end times (in timesteps) in an array and the indizes of the S file which
        trajectories have a transition; ffs is the last timestep where the traj is outside
        the bounds and ffe is the first timestep where the traj is outside the bounds again

    Raises:
        Exception: no transition detected. Check traj files and boundaries
        AttributeError: 'list' object has no attribute 'astype' if the list of starting times ffs/ffe/indizes is empty and no passage was detected; check boundaries and trajectories

    """
    # TODO avoid using for loops and instead use numpy methods
    number_of_traj = T[:, 0, 0].size
    number_of_timesteps = T[0, :, 0].size
    label = np.zeros(number_of_traj)  # how is the object labeled
    middle_count = np.zeros(number_of_traj)  # how long in the middle
    lower_count = np.zeros(number_of_traj)  # how long in the lower layer
    upper_count = np.zeros(number_of_traj)  # how long in the upper layer
    full_flips_start = []
    full_flips_end = []
    indizes = []
    for t in range(number_of_timesteps):
        for a in range(number_of_traj):
            curr = T[a, t]
            # print(t, curr, label[a], label_count[a], middle_count[a])
            if isAtomBelow(curr):  # object is below lower bound
                lower_count[a] = lower_count[a] + 1  # one time step longer in the layer
                upper_count[a] = 0  # set count of upper layer to 0
                if lower_count[a] == p:
                    if label[a] == 4:  # if it comes from above
                        full_flips_start = np.append(
                            full_flips_start, t - middle_count[a] - p
                        )
                        # end is the current timestep -p, beccause it alredy has been in the layer p steps before;
                        full_flips_end = np.append(full_flips_end, t - p + 1)
                        indizes = np.append(indizes, a)
                    label[a] = 1  # label, that its now in the lower layer
                    # set middle count to 0 (only in this if branch, because middle count
                    # (time count) should
                    # go on if the object only slips out the middle for less than p timesteps)
                    middle_count[a] = 0

            elif isAtomAbove(curr):
                upper_count[a] = upper_count[a] + 1
                lower_count[a] = 0
                if upper_count[a] == p:
                    if label[a] == 2:
                        full_flips_start = np.append(
                            full_flips_start, t - middle_count[a] - p
                        )
                        full_flips_end = np.append(full_flips_end, t - p + 1)
                        indizes = np.append(indizes, a)
                    label[a] = 5
                    middle_count[a] = 0

            # if not (isAtomBelow(curr) or isAtomAbove(curr)):
            else:
                # one timestep longer in the middle
                middle_count[a] = middle_count[a] + 1
                lower_count[a] = 0
                upper_count[a] = 0
                if middle_count[a] == p_middle:
                    # if its ready to be counted as beeing in the middle
                    if label[a] == 1:
                        label[a] = 2  # label as coming from below
                    if label[a] == 5:
                        label[a] = 4  # label as coming from above

    if len(indizes) == 0:
        raise Exception("no transition detected. Check traj files and boundaries")

    # return the start and end times of all passages that meet the conditions
    # that are set by defining p and p_middle
    return full_flips_start.astype(int), full_flips_end.astype(int), indizes.astype(int)


def bin(A, lower_bound, upper_bound):
    """
    bin trajectory file A to the boundaries (replace coordinates by the three labels 1,2,3)
    label 1 means, its below the lower bound in that timestep
    label 2 means, its between the lower and upper bound
    label 3 means, its above the upper bound

    Args:
        A (np.ndarray): trajectories (from the xvg file, but without the time-column; each column is a trajectory)
        lower_bound (float): lower bound
        upper_bound (float): upper bound

    Returns:
        np.ndarray: trajectories with the labels 1, 2, 3
    """
    number_of_traj = A[:, 0].size
    number_of_timesteps = A[0, :].size
    C = A.copy()
    # print("lower / upper bound: " +  str(l) + " / " + str(u))
    for i in range(number_of_timesteps):
        for j in range(number_of_traj):
            if A[j, i] < lower_bound:
                C[j, i] = 1
            if A[j, i] > upper_bound:
                C[j, i] = 3
            if (A[j, i] >= lower_bound) and (A[j, i] <= upper_bound):
                C[j, i] = 2
    # C[0,:]=A[0,:]
    return C


def save_1darr_to_txt(arr: np.ndarray, path: str):
    """
    Save a NumPy array to a text file.

    Args:
        arr (numpy.ndarray): The array to be saved.
        path (str): The path to the output text file including the extension .txt.

    Returns:
        None
    """
    try:
        with open(path, "w") as f:
            for i in range(0, arr.size):
                f.write("\n")
                f.write(str(arr[i]))
    except PermissionError:
        print(
            f"PermissionError: Permission denied to {path}. The results will not be saved."
        )


def fit_diffusion_pdf(L: float, passage_times: list, D_guess: float) -> float:
    """
    calculate diffusion using Gotthold Fläschner Script.

    Args:
        L: length of the membrane in Angstrom
        passage_times: passage times in ns
        D_guess: initial guess for the diffusion coefficient

    Returns:
        D_hom: diffusion coefficient calculated using PDF fit
    """
    print(f"Calculating diffusion coefficient using a PDF fit ...")
    ecdf = ECDF(passage_times)

    # PREPARE DATA
    idx = (np.abs(ecdf.y - 0.5)).argmin()
    centertime = ecdf.x[idx]
    bins = int(10 * np.max(passage_times) / centertime)
    histo, edges = np.histogram(passage_times, bins, density=True)
    center = edges - (edges[2] - edges[1])
    center = np.delete(center, 0)
    edges = np.delete(edges, 0)

    params_hom = fitting_hom_lsq(center[0:], histo[0:], L, D_guess)
    D_hom = params_hom[0]
    return D_hom


def fit_diffusion_cdf(L: float, passage_times: list, D_guess: float) -> float:
    print(f"Calculating diffusion coefficient using a CDF fit ...")
    ecdf = ECDF(passage_times)
    params_hom_cdf = fitting_hom_cdf_lsq(ecdf.x[1:], ecdf.y[1:], L)
    D_hom_cdf = params_hom_cdf[0]
    return D_hom_cdf


#########################################################################################
# START Functions from Gotthold Fläschners script #######################################
# These functions implement a PDF and CDF fit for fitting the first passage time approach
# to the distribution of first passage times of molecules through a membrane.
# The first passage time approach is described in the paper by van Hijkoop et al.
# (https://doi.org/10.1063/1.2761897)
def hom_cdf(x, D, i, L):
    t = (L) ** 2 / (i**2 * np.pi**2 * D)  # L^2/(i^2*pi^2*D)
    return (-1) ** (i - 1) * np.exp(-x / t)  # summand in Gl. 10 vanHijkoop


def fitfunc_hom_cdf(x, D, L):
    i = 50  # Summe geht bis 50 (approx statt undendlich)
    result = 0
    for j in range(1, i):
        result = result + hom_cdf(x, D, j, L)
    return 1 - 2 * result  # gleichung 10 in vanHijkoop paper


def fitfunc_hom_cdf_lsq(L):
    def f(D, x, y):
        i = 50
        result = 0
        for j in range(1, i):
            result = result + hom_cdf(x, D, j, L)
        return 1 - 2 * result - y

    return f


def fitting_hom_cdf_lsq(x_data, y_data, L):
    res_robust = least_squares(
        fitfunc_hom_cdf_lsq(L),
        x0=20,
        loss="soft_l1",
        f_scale=0.3,
        args=(x_data, y_data),
    )
    return res_robust.x


def hom(x, D, i, L):
    t = (L) ** 2 / (i**2 * np.pi**2 * D)
    return (-1) ** (i - 1) * i**2 * np.exp(-x / t)  # summand in equation 9 vanHijkoop


def fitfunc_hom(x, D, L):
    i = 151  # not to infinity but to 151 (approximation of the sum)
    result = 0
    for j in range(1, i):
        result = result + hom(x, D, j, L)
    return 2 * np.pi**2 * D / (L) ** 2 * result  # sum of equation 9 vanHijkoop


def fitfunc_hom_lsq(L):
    def f(D, x, y):
        n = 151
        sum = 0
        # print("start sum")
        for j in range(1, n):
            sum = sum + hom(x, D, j, L)
            # print(np.sum(hom(x, D, j, L)))
        eq = 2 * np.pi**2 * D / (L) ** 2 * sum
        residuals = eq - y
        return residuals  # difference vector between the fit using D and the data y

    return f


def fitting_hom_lsq(x_data, y_data, L, D0):
    # least squares depends on an initial value for D which has to be provided
    # in order to find the right global minimum. It has shown that the PDF fit
    # errors have not just one global min but several local minima. Therefore
    # the initial value is required as a user input for the least squares fit.
    # basin hopping is not suitable since it also depends on the initial value
    # and the global minimum is not found in all cases.
    # dual annealing also did not work since the bound for D is infinity on the
    # positive axis.
    res_robust = least_squares(
        fitfunc_hom_lsq(L), x0=D0, loss="soft_l1", args=(x_data, y_data), f_scale=0.3
    )
    return res_robust.x


# END Functions from Gotthold Fläschners script #########################################
#########################################################################################
