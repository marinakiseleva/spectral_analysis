"""
Runs variational inference on the model to estimate the posterior p(m,D|d)
"""

from functools import partial
import multiprocessing


from scipy.stats import multivariate_normal
from collections import OrderedDict
import numpy as np
import math


from hapke_model import get_r_mixed_hapke_estimate
from constants import c_wavelengths, pure_endmembers


def sample_dirichlet(x):
    """
    Sample from dirichlet
    :param x: Vector that will be multiplied by constant and used as alpha parameter
    """
    c = 10
    # Threshold x values so that they are always valid.
    for index, value in enumerate(x):
        if value < 0.001:
            x[index] = 0.01
    return np.random.dirichlet(alpha=x * c)


def sample_multivariate(mean):
    """
    Sample from 0-mean multivariate Gaussian (with identity matrix as covariance)
    :param mean: vector of mean of Gaussian
    """
    length = mean.shape[0]
    covariance = np.zeros((length, length))
    np.fill_diagonal(covariance, 1)

    return np.random.multivariate_normal(mean, covariance)


def get_m_prob(M, A=None):
    """
    Get probability of x from prior Dirichlet PDF on mineral abundance:
    1/B(a) * prod m_i^(a_i-1)
    :param M: vector of mineral abundances, sum to 1
    """
    def get_B(A):
        """
        Get multinomial beta function value given vector A of concentration parameters
        """
        numerator = 1
        for a in A:
            numerator *= math.gamma(a)
        denominator = math.gamma(np.sum(A))
        return numerator / denominator
    if A is None:
        A = np.array([1] * len(M))
    f = 1 / get_B(A)
    running_prod = 1
    for index, m in enumerate(M):
        running_prod *= m**(A[index] - 1)

    return f * running_prod


def get_D_prob(X):
    """
    Get probability of x from prior PDF on grain size:
    p(x) = 1 / (b-a)
    :param X: vector grain size
    """
    min_grain_size = 25
    max_grain_size = 50
    return 1 / (max_grain_size - min_grain_size)


def get_log_likelihood(d, m, D):
    """
    p(d|m, D)
    Get likelihood of reflectance spectra d, given the mineral assemblage and grain size.
    :param d: Spectral reflectance data, as Numpy vector
    :param m: Dict from SID to abundance
    :param D: Dict from SID to grain size
    """
    r_e = get_r_mixed_hapke_estimate(m, D)
    length = len(c_wavelengths)

    covariance = np.zeros((length, length))
    np.fill_diagonal(covariance, 0.01)  # 5 * (10 ** (-4))

    y = multivariate_normal.pdf(x=d, mean=r_e, cov=covariance)

    # Threshold min values to not overflow
    if y < 10**-310:
        y = 10**-310

    return math.log(y)


def transition_model(cur_m, cur_D):
    """
    Sample new m and D
    :param cur_m: Vector of mineral abundances
    :param cur_D: Vector of grain sizes
    """
    new_m = sample_dirichlet(cur_m)
    new_D = sample_multivariate(cur_D)
    return new_m, new_D


def convert_arr_to_dict(values):
    """
    Convert Numpy array to dict
    """
    d = OrderedDict()
    for index, v in enumerate(values):
        d[pure_endmembers[index]] = v
    return d


def get_log_posterior_estimate(d, m, D):
    """
    Get estimate of posterior in log.
    log p(d|m, D) + log p(m) + log p(D)
    """
    m_prior = math.log(get_m_prob(m))
    D_prior = math.log(get_D_prob(D))
    m_dict = convert_arr_to_dict(m)
    D_dict = convert_arr_to_dict(D)
    ll = get_log_likelihood(d, m_dict, D_dict)
    return ll + m_prior + D_prior


def infer_datapoint(iterations, d):
    """
    Run metropolis algorithm (MCMC) to estimate m and D
    :param iterations: Number of iterations to run over
    :param d: 1 spectral sample (1D Numpy vector)
    """
    cur_m = np.array([.33] * 3)
    cur_D = np.array([30] * 3)

    for i in range(iterations):
        # Determine whether or not to accept the new parameters, based on the
        # ratio of log (likelihood*priors)
        new_m, new_D = transition_model(cur_m, cur_D)

        cur = get_log_posterior_estimate(d, cur_m, cur_D)
        new = get_log_posterior_estimate(d, new_m, new_D)

        ratio = new / cur

        u = np.random.uniform(0, 1)
        if ratio > u:
            cur_m = new_m
            cur_D = new_D
    return [cur_m, cur_D]


def infer_image(iterations, image):
    """
    Infer m and D for entire image
    :param iterations: Number of MCMC iterations to run for each datapoint 
    :param image: 3D Numpy array with 3d dimension equal to len(c_wavelengths)
    """

    num_rows = image.shape[0]
    num_cols = image.shape[1]
    # Mineral assemblage predictions
    m_image = np.ones((num_rows, num_cols, 3))
    # Grain size predictions
    D_image = np.ones((num_rows, num_cols, 3))

    # For clarity: create map from numeric index to X,Y coords in image
    index = 0
    index_coords = {}  # List index to image index
    r_space = []  # List of reflectances for image
    for i in range(num_rows):
        for j in range(num_cols):
            index_coords[index] = [i, j]
            r_space.append(image[i, j])
            index += 1
    print("Done indexing image. Starting processing...")

    pool = multiprocessing.Pool()

    # Pass in parameters that don't change for parallel processes (# of iterations)
    func = partial(infer_datapoint, iterations)

    m_and_Ds = []
    # Multithread over the pixels' reflectances
    m_and_Ds = pool.map(func, r_space)

    pool.close()
    pool.join()
    print("Done processing...")

    # pool.map results are ordered - save them in image format
    for index, pair in enumerate(m_and_Ds):
        # retrieve x, y coords
        [i, j] = index_coords[index]
        m, D = pair
        m_image[i, j] = m
        D_image[i, j] = D

    return m_image, D_image
