"""
Find best constants (Dirichlet scaling factor and multivariate diagonal value) for sample point
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from functools import partial
from model.inference import *
from model.hapke_model import get_USGS_r_mixed_hapke_estimate
from utils.access_data import *
from utils.constants import *


def get_rmse(a, b):
    # Print error
    return math.sqrt(np.mean((a - b)**2))


def print_error(m_actual, D_actual, m_est, D_est):
    m_rmse = str(round(get_rmse(m_actual, m_est), 2))
    print("RMSE for m: " + m_rmse)
    D_rmse = str(round(get_rmse(D_actual, D_est), 2))
    print("RMSE for D: " + D_rmse)


def infer_point(r_actual, iterations, pair):
    """
    Use pixel-independent model to infer mineral assemblages and grain sizes of pixels in image
    """
    C, V = pair
    est_m, est_D = infer_datapoint(iterations=iterations, d=r_actual)

    wavelengths = N_WAVELENGTHS
    r_est = get_USGS_r_mixed_hapke_estimate(convert_arr_to_dict(est_m),
                                            convert_arr_to_dict(est_D))
    fig, ax = plt.subplots(1, 1, constrained_layout=True,
                           figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    ax.plot(wavelengths, r_est, label="Estimated")
    ax.plot(wavelengths, r_actual, label="Actual")

    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Reflectance")
    ax.legend()

    plt.ylim((0, 1))

    rmse = get_rmse(r_actual, r_est)
    print("C: " + str(C) + ", V: " + str(V) + " RMSE : " + str(rmse))

    plt.savefig("../output/figures/opt/C_" + str(C) + "_V_" + str(V) + ".png")


if __name__ == "__main__":
    """
    Find best constants for this sample point
    """

    m_random = np.array([0, 1.0, 0, 0, 0, 0, 0])
    D_random = np.array([80, 80, 60, 60, 60, 60, 60])
    true_m = convert_arr_to_dict(m_random)
    true_D = convert_arr_to_dict(D_random)
    r_actual = get_USGS_r_mixed_hapke_estimate(m=true_m, D=true_D)

    iterations = 800
    sample_Cs = [1, 2, 5, 10, 20, 30, 50, 100]
    sample_Vs = [1, 2, 5, 10]

    pairs = []
    for c in sample_Cs:
        for v in sample_Vs:
            pairs.append([c, v])

    pool = multiprocessing.Pool(NUM_CPUS)

    # Pass in parameters that don't change for parallel processes (# of iterations)
    func = partial(infer_point, r_actual, iterations)

    pool.map(func, pairs)

    pool.close()
    pool.join()
    print("Done processing...")
