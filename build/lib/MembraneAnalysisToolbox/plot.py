import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


def plot_1dtraj(Y):
    """plot Y as a trajectory

    Args:
        Y (array): 1d array containing the trajectory.
        E.g. an array containing the z-coordinates for every timestep.
    """
    plt.plot(np.arange(Y.size), np.transpose(Y))


def plot_hor_bounds(bounds):
    """plot the boundaries as horizontal dotted lines

    Args:
        bounds (array): contains the boundaries. E.g. [10,30].
    """
    for b in bounds:
        plt.axhline(y=b, linestyle="--")


def plot_point(x, y):
    """plot the point at (x,y) as a red point

    Args:
        x (float): x-coordinate of the point
        y (float): y-coordinate of the point
    """
    plt.plot(x, y, "ro")


def plot_dist(A, max_range=500, number_of_bins=100):
    """plots the histrogram of the entries in A.
    The bars are normed so that the highest bar has value 1.

    Args:
        A (array): contains the data. E.g. passage times
        max_range (int, optional): how far the histogram reaches. Defaults to 500.
        number_of_bins (int, optional): how many bins for the histogram. Defaults to 100.
    """
    width_of_bins = max_range / number_of_bins
    H, b = np.histogram(A, bins=number_of_bins, range=(0, max_range), density=1)
    H = H * (1 / H.max())  # normalization
    plt.bar(
        b[:-1] + width_of_bins / 2,
        H,
        width=width_of_bins,
        alpha=0.4,
        edgecolor="black",
        color="blue",
    )


def plot_3dtraj(ax, X, Y, Z):
    """plot trajectory in 3d.

    Args:
        ax (axis): 3d axis in matplotlib, that has to be created previously
        X (array): x-coordinates
        Y (array): y-coordinates
        Z (array): z-coordinates
    """
    ax.plot(X, Y, Z)


def plot_3dpoints(ax, X, Y, Z):
    """plot points in 3d

    Args:
        ax (_type_): 3d axis in matplotlib, that has to be created previously
        X (_type_): x-coordinates
        Y (_type_): y-coordinates
        Z (_type_): z-coordinates
    """
    ax.scatter(X, Y, Z)


def plot_3dbounds(ax, bounds):
    """plot 3d bundaries as layers in a 3d plot. The layers are parralel to the x-y-layer
    and have the z-coordinate of the passed bounds array

    Args:
        ax (_type_): 3d axis in matplotlib, that has to be created previously
        bounds (_type_): z-coordinate of the boundaries layers
    """
    xx, yy = np.meshgrid(range(50), range(50))
    for b in bounds:
        ax.plot_surface(xx, yy, xx * 0 + b, alpha=0.1)


def plot_bintraj(A, timeline, bounds):
    """plot the amount of objects below the lower, above the upper and between
    the bounds over the time.

    Args:
        A (_type_): trajectories without the time-column; each column is a trajectory
        timeline (_type_): array of floats, that contain the z-koordinate of the boundaries
        bounds (_type_): timeline is the time corresponding to the trajectories in ns (the x-axis)
    """
    number_of_timesteps = A[0, :].size
    binned = bin(A, bounds[0], bounds[1])  # bin the trajectories using bin
    number_of_1 = np.zeros(number_of_timesteps)
    number_of_2 = np.zeros(number_of_timesteps)
    number_of_3 = np.zeros(number_of_timesteps)

    for t in range(number_of_timesteps):
        number_of_1[t] = np.count_nonzero(
            binned[:, t] == 1
        )  # count the number of ones in the binned array
        number_of_2[t] = np.count_nonzero(binned[:, t] == 2)
        number_of_3[t] = np.count_nonzero(binned[:, t] == 3)

    # plot without interpolation
    # plt.plot(timeline,number_of_1, label='number in lower')
    # plt.plot(timeline,number_of_2, label='number in middle')
    # plt.plot(timeline,number_of_3, label='number in upper')

    # interpolate with scipy b-splines to make the curve smoother
    tck1 = interpolate.splrep(timeline[::100], number_of_1[::100])
    tck2 = interpolate.splrep(timeline[::100], number_of_2[::100])
    tck3 = interpolate.splrep(timeline[::100], number_of_3[::100])
    plt.plot(timeline, interpolate.splev(timeline, tck1, der=0), label="lower")
    plt.plot(timeline, interpolate.splev(timeline, tck2, der=0), label="middle")
    plt.plot(timeline, interpolate.splev(timeline, tck3, der=0), label="upper")
