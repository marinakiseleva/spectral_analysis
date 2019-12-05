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


def get_reflectance_error(wavelength, r, n, k, D):
    """
    Gets difference between estimated reflectance using Hapke model and reflectance in data, for a wavelength at a certain index
    :param wavelength: wavelength 
    :param r: reflectance at wavelength index, scalar
    :param n: optical constant n, scalar (constant across all wavelengths)
    :param k: optical constnat k, scalar (for this wavelength)
    :param D: grain size, scalar
    """

    # Calculated estimated reflectance
    r_e = get_reflectance_hapke_estimate(mu, mu_0, n, k, D, wavelength)

    return get_rmse(r, r_e)


def get_D_avg_refl_error(sid, sample_spectra, index, k):
    """
    Gets average reflectance error over range of grain sizes
    :param sid: Spectrum ID
    :param sample_spectra: Pandas DataFrame of data
    :param index: Index of wavelength to check error on
    :param k: k value to evaluate
    """

    d_min, d_max = get_grain_sizes(sid, sample_spectra)
    grain_sizes = list(range(int(d_min), int(d_max)))

    # Get wavelength at wavelength index
    wavelength = get_wavelengths(sid, sample_spectra)[index]
    # Get actual reflectance spectra from data, for this wavelength index
    r = get_reflectance_spectra(sid, sample_spectra)[index]

    n = sids_n[sid]

    rmses = []
    for D in grain_sizes:
        rmse = get_reflectance_error(wavelength, r, n, k, D)
        rmses.append(rmse)
    return sum(rmses) / len(rmses)


def get_best_k(sid, sample_spectra):
    min_ks = []
    min_k_errors = []

    for index, l in enumerate(c_wavelengths):
        print("Getting best k error for SID: " + str(sid) + ", on index " + str(index))

        # 100,000 values from 10^-14 to 1, in log space
        #k_space = np.logspace(-14, -1, 1e5)
        k_space = np.logspace(-14,-1,1000)
        pool = multiprocessing.Pool()
        l_errors = []
        func = partial(get_D_avg_refl_error, sid, sample_spectra, index)
        # Multithread over different values in the k space
        l_errors = pool.map(func, k_space)
        pool.close()
        pool.join()

        min_index = l_errors.index(min(l_errors))
        min_k = k_space[min_index]

        min_ks.append(min_k)
        min_k_errors.append(min(l_errors))

    return np.array(min_ks), np.array(min_k_errors)


######################################################
# Helper functions
######################################################
def get_mu_0(sample_row):
    """
    Gets mu_0, cosine of incidence angle 
    :param sample_row: Pandas series from sample spectra DataFrame
    """
    source_angle = sample_row["SourceAngle"].values[0]
    return get_cosine(source_angle)


def get_mu(sample_row):
    """
    Gets mu, cosine of detect_angle emission/emergence angle
    :param sample_row: Pandas series from sample spectra DataFrame
    """
    detect_angle = sample_row["DetectAngle"].values[0]
    return get_cosine(detect_angle)


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
