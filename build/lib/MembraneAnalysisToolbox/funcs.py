import numpy as np


def calc_hor_dist(x_traj, y_traj, ffs, ffe):
    """calculate the horizontal x-y-distance that the molecule goes
    when going through the membrane

    Args:
        x_traj (_type_): _description_
        y_traj (_type_): _description_
        ffs (_type_): _description_
        ffe (_type_): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    """
    x_traj and y_traj are all the x and y trajectories of the passages
    ffs and ffe are the start and endimes of the transitions
    """
    # print("ooooooo")
    distances = []
    for i in range(x_traj.shape[0]):  
        # Shape[0] returns size of first dimension (number of trajs)
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
    x_traj and y_traj are all the x and y trajectories of transitions
    ffs and ffe are the start and endimes of the transitions

    returns indizes of direct transitions
    """
    d = []
    # cutoff ab wann man definiert, dass nen teilchen sein direkten weg verlassen hat
    # TODO: avaoid hard coding 4.5
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


# TODO make it more effficient (matrix operations of numpy to simplyfy the process?)
def dur_dist_improved(S, bounds, p=1, p_middle=1):
    """measure start and endpoint of passages through the bounds.
    returns the array of starting times and the array of endtimes (not in ns, but in timesteps!)

    Args:
        S (_type_): trajectories without the time-column; each column is a trajectory
        bounds (_type_): array of floats, that contains the z-coordinate of the boundaries
        p (int, optional): timesteps, that an object has to be above or below a
        bound to be seen as above or below. different values than 1 can make sense
        to compensate uncontinuous behavior (see documentation for more details). Defaults to 1.
        p_middle (int, optional): timesteps, that an object has to be in the middle to
        be seen as passaging through the middel. 3 means, the object has to be in
        the middle for 3 timesteps to be counted as a valid transition. Defaults to 1.

    Returns:
        _type_: flip start und end times in an array and the indizes of the S file which
        trajectories have a transition; ffs is the last timestep where the traj is outside
        the bounds and ffe is the first timestep where the traj is outside the bounds again


    errors:     if this method throws the error "AttributeError: 'list' object has no attribute
    'astype'" it means, that the list of starting times ffs/ffe/indizes is empty and no
    passage was detected. Check boundaries and trajectories!
    """
    number_of_traj = S[:, 0].size
    number_of_timesteps = S[0, :].size
    label = np.zeros(number_of_traj)  # how is the object labeled
    middle_count = np.zeros(number_of_traj)  # how long in the middle
    lower_count = np.zeros(number_of_traj)  # how long in the lower layer
    upper_count = np.zeros(number_of_traj)  # how long in the upper layer
    full_flips_start = []
    full_flips_end = []
    indizes = []
    for t in range(number_of_timesteps):
        for a in range(number_of_traj):
            curr = S[a, t]
            # print(t, curr, label[a], label_count[a], middle_count[a])
            if curr < bounds[0]:  # object is below lower bound
                lower_count[a] = lower_count[a] + 1  # one time step longer in the layer
                if (
                    lower_count[a] == p
                ):  # the timestep when it will be labeled with layer 1 (lower layer)
                    if label[a] == 4:  # if it comes from above
                        # start (first entry to middle layer) start time should be still outside
                        full_flips_start = np.append(
                            full_flips_start, t - middle_count[a] - p
                        )
                        # end is the current timestep -p, beccause it alredy has been in the
                        # layer p steps before;
                        # +1 to have the end time already outside the layer
                        full_flips_end = np.append(full_flips_end, t - p + 1)
                        indizes = np.append(indizes, a)
                    label[a] = 1  # label, that its now in the lower layer
                    # set middle count to 0 (only in this if branch, because middle count
                    # (time count) should
                    # go on if the object only slips out the middle for less than p timesteps)
                    middle_count[a] = 0
                upper_count[a] = 0  # set count of upper layer to 0

            if curr < bounds[1] and curr >= bounds[0]:  # object is between boundaries
                middle_count[a] = (
                    middle_count[a] + 1
                )  # one timestep longer in the middle
                if (
                    middle_count[a] == p_middle
                ):  # if its ready to be counted as beeing in the middle
                    if label[a] == 1:
                        label[a] = 2  # label as coming from below
                    if label[a] == 5:
                        label[a] = 4  # label as coming from above
                lower_count[a] = 0
                upper_count[a] = 0

            if (
                curr >= bounds[1]
            ):  # object is above upper layer (analog procedure to below lower bound)
                upper_count[a] = upper_count[a] + 1
                if upper_count[a] == p:
                    if label[a] == 2:
                        full_flips_start = np.append(
                            full_flips_start, t - middle_count[a] - p
                        )
                        full_flips_end = np.append(full_flips_end, t - p + 1)
                        indizes = np.append(indizes, a)
                    label[a] = 5
                    middle_count[a] = 0
                lower_count[a] = 0

    # TODO raise exeption if no passage has been detected for better understanding of errors
    if len(indizes) == 0:
        raise Exception(
                    "no transition detected. Check traj files and boundaries"
                )

    # return the start and end times of all passages that meet the conditions
    # that are set by defining p and p_middle
    return full_flips_start.astype(int), full_flips_end.astype(int), indizes.astype(int)


def bin(A, lower_bound, upper_bound):
    """
    bin trajectory file A to the boundaries (replace coordinates by the three labels 1,2,3)
    label 1 means, its below the lower bound in that timestep
    label 2 means, its between the lower and upper bound
    label 3 means, its above the upper bound

    A:  trajectories (from the xvg file, but without the time-column; each column is a trajectory)
    l:  lower bound
    u:  upper bound
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


def npy2txt_save(ff, path):
    """convert ff array (all passage times) saved in a .npy file by the function save_results
    to a txt file with all passage times as lines
    dont forget to rename the text file with temperature, length and number

    Args:
        ff (_type_): numpy array
        path (_type_): path to save destination
        example: path = "./diffusion/solv-hex/tt-ns_solv-hex_L181_310K_N283.txt"
    """

    # print(ff)

    n = ff.size
    # only use every nth passage time, to get around 2000 passage times, more is not
    # needed for diffusion fitting
    nth = max(int(n / 2000), 1)

    # save to txt at the given location
    with open(path, "w") as f:
        for i in range(0, ff.size, nth):
            f.write("\n")
            f.write(str(ff[i]))
