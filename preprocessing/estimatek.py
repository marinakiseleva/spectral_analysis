#!/usr/bin/env python
# coding: utf-8

# Estimate optical constant k. Does either for USGS-exact wavelengths
# or USGS reduced to CRISM wavelenghts (which vary by image)
#
import os
import pickle
import math
import pandas as pd
import numpy as np


from functools import partial
import multiprocessing

from model.hapke_model import get_reflectance_hapke_estimate
from utils.access_data import *
from utils.constants import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


######################################################
# RELAB data estimation
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
    r_e = get_reflectance_hapke_estimate(n, k, D, wavelength)
    return get_rmse(r, r_e)


def get_RELAB_D_avg_refl_error(grain_sizes, wavelengths, reflectance, sid, spectra_db, index, k):
    """
    Gets average reflectance error over range of grain sizes for RELAB spectral sample
    :param sid: Spectrum ID
    :param spectra_db: Pandas DataFrame of RELAB data
    :param index: Index of wavelength to check error on
    :param k: k value to evaluate
    """

    # Get wavelength at wavelength index
    wavelength = wavelengths[index]
    # Get actual reflectance spectra from data, for this wavelength index
    r = reflectance[index]
    n = ENDMEMBERS_N[sid]

    rmses = []
    for D in grain_sizes:
        rmse = get_reflectance_error(wavelength, r, n, k, D)
        rmses.append(rmse)
    return sum(rmses) / len(rmses)


def get_best_RELAB_k(sid, spectra_db):
    """
    Estimate best imaginary optical constant k for the reflectance of this endmember
    :param sid: Spectrum ID
    :param spectra_db: Pandas DataFrame of RELAB data
    """
    min_ks = []
    min_k_errors = []

    d_min, d_max = get_grain_sizes(sid, spectra_db)
    grain_sizes = list(range(int(d_min), int(d_max)))

    wavelengths = get_RELAB_wavelengths(sid, spectra_db, True)
    reflectance = get_reflectance_spectra(sid, spectra_db, True)

    # Only keep rows with reduced wavelengths
    with open(MODULE_DIR + "/utils/FILE_CONSTANTS/RW_BASALT.pickle", 'rb') as handle:
        RW_BASALT = pickle.load(handle)

    # Parallelize over wavelengths
    for index, l in enumerate(RW_BASALT):
        print("Getting best k error for SID: " + str(sid) + ", on index " + str(index))

        # 100,000 values from 10^-14 to 1, in log space
        k_space = np.logspace(-14, -1, 1000)
        pool = multiprocessing.Pool(8)

        l_errors = []
        func = partial(get_RELAB_D_avg_refl_error, grain_sizes,
                       wavelengths, reflectance, sid, spectra_db, index)
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
# USGS data estimation
######################################################

def get_endmember_k_RMSE(data, grainsize, wavelength, index, n, k):
    """ 
    Get reflectance error between given refletance and derived reflectance using Hapke
    :param data: reflectance (length of wavelengths)
    :param n: Optical constant n
    :param k: k value to evaluate
    """
    r_e = get_reflectance_hapke_estimate(
        n=n,
        k=k,
        D=grainsize,
        wavelengths=wavelength)
    return get_rmse(data[index], r_e)


def get_best_USGS_k(endmember, CRISM_match=False):
    """
    Estimate best imaginary optical constant k for the reflectance of this endmember
    :param endmember: Endmember name 
    """
    # get endmember reflectance
    reflectance = get_USGS_preprocessed_data(endmember, CRISM_match)
    wavelengths = get_USGS_wavelengths(CRISM_match)
    grainsize = USGS_GRAIN_SIZES[endmember]
    n = ENDMEMBERS_N[endmember]

    ks = []  # k per wavelength
    error = 0  # overall RMSE
    # Get best k per wavelength
    for index, wavelength in enumerate(wavelengths):
        print(str(index) + " / " + str(len(wavelengths)))

        # X values from 10^-14 to 1, in log space
        X = 10000
        k_space = np.logspace(-14, -1, X)
        pool = multiprocessing.Pool(NUM_CPUS)

        rmses = []
        func = partial(get_endmember_k_RMSE,
                       reflectance,
                       grainsize,
                       wavelength,
                       index,
                       n)
        # Multithread over different values in the k space
        rmses = pool.map(func, k_space)
        pool.close()
        pool.join()

        min_index = rmses.index(min(rmses))
        min_k = k_space[min_index]
        ks.append(min_k)
        error += rmses[min_index]

    RMSE = error / len(wavelengths)
    return ks, RMSE


def estimate_all_USGS_k(CRISM_match):
    """
    Estimate k for all endmembers from USGS
    """
    print("\nGoing to save in K_DIR = " + K_DIR + "\n")
    if not os.path.exists(K_DIR):
        os.makedirs(K_DIR)
    members = ["diopside",
               "augite",
               "pigeonite",
               "hypersthene",
               "enstatite",
               "andesine",
               "labradorite",
               "olivine (Fo51)",
               "magnetite"]

    for endmember in members:
        print("\n\nEstimating " + endmember)
        ks, RMSE = get_best_USGS_k(endmember, CRISM_match)
        print(str(endmember) + " length: " + str(len(ks)))
        print("Found with error " + str(RMSE))
        save_USGS_endmember_k(endmember, ks)


######################################################
# Helper functions
################################################

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

    print("Matching CRISM-USGS wavelengths.")
    estimate_all_USGS_k(True)
