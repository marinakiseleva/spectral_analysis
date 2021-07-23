"""
Runs variational inference on the model to estimate the posterior p(m,D|d)
"""
import sys
from datetime import datetime
from functools import partial
import multiprocessing
import numpy as np
import math
from scipy.stats import multivariate_normal
from collections import OrderedDict

from model.hapke_model import get_USGS_r_mixed_hapke_estimate
from utils.constants import *


def sample_dirichlet(x, C):
    """
    Sample from dirichlet
    :param x: Vector that will be multiplied by constant and used as alpha parameter
    """
    # Dirichlet sampling
    # Threshold x values so that they are always valid.
    for index, value in enumerate(x):
        if value < 0.0001:
            x[index] = 0.001
    new_x = np.random.dirichlet(alpha=x * C)
    return new_x


def m_transition(x, C):
    """
    Sample from 0-mean multivariate, add this to m assemblage vector. Then ensure they all sum to 1.
    :param x: Vector of mineral assemblage
    """
    new_x = sample_dirichlet(x, C)

    return new_x


def D_transition(mean, covariance):
    """
    Sample from 0-mean multivariate Gaussian (with identity matrix as covariance)
    :param mean: vector of mean of Gaussian
    """
    covariance_matrix = np.zeros((USGS_NUM_ENDMEMBERS, USGS_NUM_ENDMEMBERS))
    np.fill_diagonal(covariance_matrix, covariance)
    sample = np.random.multivariate_normal(mean, covariance_matrix)
    for index, v in enumerate(sample):
        if v > GRAIN_SIZE_MAX:
            sample[index] = GRAIN_SIZE_MAX
        elif v < GRAIN_SIZE_MIN:
            sample[index] = GRAIN_SIZE_MIN
        else:
            sample[index] = int(v)
    return sample


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
    return 1 / (GRAIN_SIZE_MAX - GRAIN_SIZE_MIN)


def get_likelihood(d, m, D):
    """
    p(d|m, D)
    Get likelihood of reflectance spectra d, given the mineral assemblage and grain size.
    :param d: Spectral reflectance data, as Numpy vector
    :param m: Dict from SID to abundance
    :param D: Dict from SID to grain size
    """
    r_e = get_USGS_r_mixed_hapke_estimate(m, D)
    length = len(d)
    covariance = np.zeros((length, length))
    np.fill_diagonal(covariance,  0.01)  # 5 * (10 ** (-4)))

    y = multivariate_normal.pdf(x=d, mean=r_e, cov=covariance)

    # Threshold min values to not overflow
    if y < 10**-10:
        y = 10**-10

    return y


def get_log_likelihood(d, m, D):
    """
    log (p(d|m, D))
    Get log likelihood of reflectance spectra d, given the mineral assemblage and grain size.
    :param d: Spectral reflectance data, as Numpy vector
    :param m: Dict from SID to abundance
    :param D: Dict from SID to grain size
    """
    return math.log(get_likelihood(d, m, D))


def transition_model(cur_m, cur_D, V, C):
    """
    Sample new m and D
    :param cur_m: Vector of mineral abundances
    :param cur_D: Vector of grain sizes
    :param V: covariance diagonal for grain size, D
    :param C: scaling factor for sampling mineral assemblage from Dirichlet, m
    """
    new_D = D_transition(cur_D, V)
    new_m = m_transition(cur_m, C)
    return new_m, new_D


def convert_USGS_arr_to_dict(values):
    """
    Converts Numpy array of proportions to dictionary from endmember name to proportion
    """
    d = OrderedDict()
    for index, v in enumerate(values):
        d[USGS_PURE_ENDMEMBERS[index]] = v
    return d


def get_log_posterior_estimate(d, m, D):
    """
    Get estimate of posterior in log : log p(m, D|d)
    log p(d|m, D) + log p(m) + log p(D)

    Omit priors because they are uniform
    """
    # m_prior = math.log(get_m_prob(m))
    # D_prior = math.log(get_D_prob(D))
    m_dict = convert_USGS_arr_to_dict(m)
    D_dict = convert_USGS_arr_to_dict(D)
    ll = get_log_likelihood(d, m_dict, D_dict)
    return ll  # + m_prior + D_prior


def get_posterior_estimate(d, m, D):
    """
    Get estimate of posterior
    p(m,D|d) = p(d|m, D) p(m) p(D)

    Omit priors because they are uniform
    """
    # m_prior = get_m_prob(m)
    # D_prior = get_D_prob(D)
    m_dict = convert_arr_to_dict(m)
    D_dict = convert_arr_to_dict(D)
    ll = get_likelihood(d, m_dict, D_dict)
    return ll  # * m_prior * D_prior


def infer_datapoint(d_seeds_index, iterations, V, C):
    """
    Run metropolis algorithm (MCMC) to estimate m and D. Return the MAP.
    :param d_seeds_index: [reflectance, seed, pixel num] (1D Numpy vector)  
    :param V: covariance diagonal for grain size, D
    :param C: scaling factor for sampling mineral assemblage from Dirichlet, m
    """
    d, seed, pixel_num = d_seeds_index
    np.random.seed(seed=seed)
    # Initialize randomly
    cur_m = sample_dirichlet(np.random.random(USGS_NUM_ENDMEMBERS), C)
    cur_D = np.full(shape=USGS_NUM_ENDMEMBERS, fill_value=INITIAL_D)
    unchanged_i = 0  # Number of iterations since last update
    MAP_mD = [cur_m, cur_D, 0]
    for i in range(iterations):
        new_m, new_D = transition_model(cur_m, cur_D, V, C)
        new_post = get_posterior_estimate(d, new_m, new_D)
        cur_post = get_posterior_estimate(d, cur_m, cur_D)

        ratio = new_post / cur_post
        phi = min(1, ratio)
        u = np.random.uniform(0, 1)
        if phi >= u:
            unchanged_i = 0
            cur_m = new_m
            cur_D = new_D
            if new_post > MAP_mD[2]:
                MAP_mD = [new_m, new_D, new_post]
        else:
            unchanged_i += 1

        if unchanged_i > INF_EARLY_STOP:
            print("\nEarly Stop at iter: " + str(i))
            break
    if pixel_num % 10 == 0:
        print(str(pixel_num) + "  datapoint finished.")
        sys.stdout.flush()

    return MAP_mD[:2]


def infer_superpixels(iterations, superpixels, V, C):
    """
    Infer m and D for each superpixel independently
    :param superpixels: List of reflectances (each is Numpy array)
    :param V: covariance diagonal for grain size, D
    :param C: scaling factor for sampling mineral assemblage from Dirichlet, m
    """
    pool = multiprocessing.Pool(NUM_CPUS)

    # Create seed for each reflectance (so that each process has a random seed.)
    # This is necessary because processes need randomness for sampling.
    d_seeds_indices = []
    for i, r in enumerate(superpixels):
        seed = np.random.randint(100000)
        d_seeds_indices.append([r, seed, i])

    func = partial(infer_datapoint, iterations=iterations, V=V, C=C)

    m_and_Ds = []
    # Multithread over the pixels' reflectances
    m_and_Ds = pool.map(func, d_seeds_indices)
    pool.close()
    pool.join()
    print("Done processing...")
    return m_and_Ds


def infer_points_parallel(iterations, V, C, d_seeds_indices):
    """
    Infer data points as pool
    """
    print("Starting parallel processing...")

    pool = multiprocessing.Pool(NUM_CPUS)

    # Pass in parameters that don't change for parallel processes
    func = partial(infer_datapoint,
                   iterations=iterations,
                   V=V,
                   C=C)
    m_and_Ds = []
    m_and_Ds = pool.imap(func, d_seeds_indices)

    pool.close()
    pool.join()
    print("Done processing...")
    return m_and_Ds


def reconstruct_image(num_rows, num_cols, index_coords, m_and_Ds):
    """
    Create 2D m and D image from list of inferred m and D 
    """

    # Mineral assemblage predictions
    m_image = np.ones((num_rows, num_cols, USGS_NUM_ENDMEMBERS))
    # Grain size predictions
    D_image = np.ones((num_rows, num_cols, USGS_NUM_ENDMEMBERS))

    # pool.map results are ordered - save them in image format
    for index, pair in enumerate(m_and_Ds):
        # retrieve x, y coords
        [i, j] = index_coords[index]
        m, D = pair
        m_image[i, j] = m
        D_image[i, j] = D
    return m_image, D_image


def cos_img(img):
    """
    Take cos of each value in image and save cos(img)
    """
    rad_img = np.radians(img)
    return np.cos(rad_img)


def infer_image(iterations, image, V, C):
    """
    Infer m and D for entire image - each pixel indepedently
    :param iterations: Number of MCMC iterations to run for each datapoint
    :param image: 3D Numpy array with 3d dimension equal to wavelengths
    :param V: covariance diagonal for grain size, D
    :param C: scaling factor for sampling mineral assemblage from Dirichlet, m
    """
    num_rows = image.shape[0]
    num_cols = image.shape[1]

    # index_coords Map from list index to X,Y coords in image
    index_coords = {}
    r_space = []  # List of reflectances for image
    index = 0
    for i in range(num_rows):
        for j in range(num_cols):
            index_coords[index] = [i, j]
            r_space.append(image[i, j])
            index += 1

    d_seeds_indices = []
    for i, r in enumerate(r_space):
        # Each parallel process needs random seed for sampling
        seed = np.random.randint(100000)
        d_seeds_indices.append([r_space[i], seed, i])

    m_and_Ds = infer_points_parallel(iterations, V, C, d_seeds_indices)

    m_image, D_image = reconstruct_image(num_rows, num_cols, index_coords, m_and_Ds)

    return m_image, D_image


def init_mrf(image, V, C):
    """
    Set random mineral & grain  assemblage for each pixel and return 3D Numpy array with 3rd dimension as assemblage
    :param image: 3D Numpy array with 3rd dimension equal to # of wavelengths
    :param V: covariance diagonal for grain size, D
    :param C: scaling factor for sampling mineral assemblage from Dirichlet, m
    """
    N = USGS_NUM_ENDMEMBERS
    covariance = np.zeros((N, N))
    np.fill_diagonal(covariance, V)

    num_rows = image.shape[0]
    num_cols = image.shape[1]
    m_image = np.zeros((num_rows, num_cols, N))
    D_image = np.zeros((num_rows, num_cols, N))
    for i in range(num_rows):
        for j in range(num_cols):
            reflectance = image[i, j]

            rand_D = D_transition(np.array([INITIAL_D] * N), V)
            rand_m = sample_dirichlet(np.array([float(1 / N)] * N), C)

            m_image[i, j] = rand_m
            D_image[i, j] = rand_D

    return m_image, D_image


def convert_arr_to_dict(values):
    """
    Convert Numpy array to dict
    """
    d = OrderedDict()
    for index, v in enumerate(values):
        d[USGS_PURE_ENDMEMBERS[index]] = v
    return d


def get_SAD(a, b):
    """
    Get spectral angle distance:
        d(i,j) =  (i^T * j) / ( ||i|| ||j|| )
    :param a: Numpy vector
    :param b: Numpy vector
    """
    n = np.dot(a.transpose(), b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return np.arccos(n / d)


def get_distance(a,  b):
    """
    Get distance between 2 mineral assemblage vectors
    """

    # Euclidean
    return np.linalg.norm(a - b)
