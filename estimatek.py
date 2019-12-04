#!/usr/bin/env python
# coding: utf-8

# Exploring data provided by RELAB
#
# Documentation available:
# http://www.planetary.brown.edu/relabdata/catalogues/Catalogue_README.html


import pandas as pd
import numpy as np
import math

from functools import partial
import multiprocessing

from hapke_model import get_reflectance_hapke_estimate
from access_data import *
from constants import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


######################################################
# Define constants
######################################################


def get_reflectance_rmse(sid, sample_spectra, k, D):
    sample_row = sample_spectra[sample_spectra['SpectrumID'] == sid]
    # source_angle : incidence angle in degrees
    source_angle = sample_row["SourceAngle"].values[0]
    mu_0 = get_cosine(source_angle)

    # detect_angle emission/emergence angle in degrees
    detect_angle = sample_row["DetectAngle"].values[0]
    mu = get_cosine(detect_angle)

    wavelengths = get_wavelengths(sid, sample_spectra)

    n = sids_n[sid]

    # Calculated estimated reflectance
    r_e = get_reflectance_hapke_estimate(mu, mu_0, n, k, D, wavelengths)

    # Get actual reflectance from data
    r = get_reflectance_spectra(sid, sample_spectra)

    return get_rmse(r, r_e)


def get_D_average_rmse(sid, sample_spectra, k):
    print("Runnign D average RMSE with params " + str(sid) + " and k " + str(k))

    d_min, d_max = get_grain_sizes(sid, sample_spectra)
    # D = d_min
    grain_sizes = list(range(int(d_min), int(d_max)))
    rmses = []
    for D in grain_sizes:
        rmse = get_reflectance_rmse(sid, sample_spectra, k, D)
        rmses.append(rmse)
    return sum(rmses) / len(rmses)


def get_best_k(sid, sample_spectra):
    # 100,000 values from 10^-14 to 1, in log space
    k_space = np.logspace(-14, -1, 1e5)
    pool = multiprocessing.Pool()
    all_rmses = []
    func = partial(get_D_average_rmse, sid, sample_spectra)
    all_rmses = pool.map(func, k_space)

    pool.close()
    pool.join()
    min_index = all_rmses.index(min(all_rmses))
    min_rmse = min(all_rmses)
    min_k = k_space[min_index]

    return min_k, min_rmse


######################################################
# Helper functions
######################################################
def get_rmse(a, b):
    """
    Get RMSE between 2 Numpy arrays
    """
    return np.sqrt(np.mean((a - b)**2))


def get_cosine(x):
    """
    Gets cosine of x 
    :param x: in degrees
    """
    return math.cos(math.radians(x))


if __name__ == "__main__":

    sample_spectra = get_data()
    olivine_k, best_rmse = get_best_k(pure_olivine_sid, sample_spectra)
    print("Best k for olivine is : " + str(olivine_k) + " with RMSE: " + str(best_rmse))

    enstatite_k, best_rmse = get_best_k(pure_enstatite_sid, sample_spectra)
    print("Best k for enstatite is : " + str(enstatite_k) + " with RMSE: " + str(best_rmse))

    anorthite_k, best_rmse = get_best_k(pure_anorthite_sid, sample_spectra)
    print("Best k for anorthite is : " + str(anorthite_k) + " with RMSE: " + str(best_rmse))
