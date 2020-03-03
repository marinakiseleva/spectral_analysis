"""
Runs variational inference on the model to estimate the posterior p(m,D|d)
"""

from functools import partial
import multiprocessing


from scipy.stats import multivariate_normal
from collections import OrderedDict
import numpy as np
import math


from model.hapke_model import get_r_mixed_hapke_estimate
from utils.constants import c_wavelengths, pure_endmembers, NUM_ENDMEMBERS, GRAIN_SIZE_MIN, GRAIN_SIZE_MAX


def sample_dirichlet(x):
    """
    Sample from dirichlet
    :param x: Vector that will be multiplied by constant and used as alpha parameter
    """
    c = 100
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
    np.fill_diagonal(covariance, .1)

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
    return 1 / (GRAIN_SIZE_MAX - GRAIN_SIZE_MIN)


def get_likelihood(d, m, D):
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
    # Initialize with 1/3 each mineral and grain size 30 for each
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
    print("Finished datapoint.")
    return [cur_m, cur_D]


def infer_segmented_image(iterations, superpixels):
    """
    Infer m and D for each superpixel 
    :param superpixels: List of reflectances (each is Numpy array)
    """
    # Use 1/4 of CPUs
    num_processes = int(multiprocessing.cpu_count() / 4)
    print("Running " + str(num_processes) + " processes.")
    pool = multiprocessing.Pool(num_processes)

    # Pass in parameters that don't change for parallel processes (# of iterations)
    func = partial(infer_datapoint, iterations)

    m_and_Ds = []
    # Multithread over the pixels' reflectances
    m_and_Ds = pool.map(func, superpixels)

    pool.close()
    pool.join()
    print("Done processing...")

    return m_and_Ds


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

    # Use 1/4 of CPUs
    num_processes = int(multiprocessing.cpu_count() / 4)
    print("Running " + str(num_processes) + " processes.")
    pool = multiprocessing.Pool(num_processes)

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


def init_gibbs(image):
    """
    Set random mineral & grain  assemblage for each pixel and return 3D Numpy array with 3rd dimension as assemblage
    :param image: 3D Numpy array with 3rd dimension equal to len(c_wavelengths)
    """
    num_rows = image.shape[0]
    num_cols = image.shape[1]
    m_image = np.zeros((num_rows, num_cols, NUM_ENDMEMBERS))
    D_image = np.zeros((num_rows, num_cols, NUM_ENDMEMBERS))
    for i in range(num_rows):
        for j in range(num_cols):
            reflectance = image[i, j]

            rand_m = sample_dirichlet(np.array([.33] * 3))
            rand_D = sample_multivariate(np.array([30] * 3))

            m_image[i, j] = rand_m
            D_image[i, j] = rand_D

    return m_image, D_image


def get_posterior_estimate(d, m, D):
    """
    Get estimate of posterior in log.
    p(m,D|d) = p(d|m, D) p(m) p(D)
    """
    m_prior = get_m_prob(m)
    D_prior = get_D_prob(D)
    m_dict = convert_arr_to_dict(m)
    D_dict = convert_arr_to_dict(D)
    ll = get_likelihood(d, m_dict, D_dict)
    return ll * m_prior * D_prior


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


def get_crf_posterior(m_image, D_image, i, j, m, D, d):
    """
    Compute
    - log(P(y_i | x_i)) + sum_{n in neighbors} SAD(y_i, y_n)
     :param m_image: 3D Numpy array, mineral assemblages for pixels
    :param D_image: 3D Numpy array, grain sizes for pixels
    :param i: row index for datapoint d
    :param j: col index for datapoint d
    :param m: Mineral assemblage for pixel i,j to consider
    :param D: Grain sizes for pixel i,j to consider
    :param d: 1 spectral sample (1D Numpy vector)
    """

    e_spectral = get_posterior_estimate(d, m, D)

    num_rows = m_image.shape[0]
    num_cols = m_image.shape[1]
    # get energy of neighbors
    n_energy = 0
    cur_row = i
    cur_col = j
    row_above = i - 1
    row_below = i + 1
    left_col = j - 1
    right_col = j + 1

    if row_above >= 0:
        # Above
        n_energy += get_SAD(m_image[row_above, j], m)
        if left_col >= 0:
            # Top left
            n_energy += get_SAD(m_image[row_above, left_col], m)
        if right_col < num_cols:
            # Top right
            n_energy += get_SAD(m_image[row_above, right_col], m)

    if row_below < num_rows:
        # Below
        n_energy += get_SAD(m_image[row_below, j], m)
        if left_col >= 0:
            # Bottom left
            n_energy += get_SAD(m_image[row_below, left_col], m)
        if right_col < num_cols:
            # Top right
            n_energy += get_SAD(m_image[row_below, right_col], m)

    if left_col >= 0:
        # Left
        n_energy += get_SAD(m_image[cur_row, left_col], m)
    if right_col < num_cols:
        # Right
        n_energy += get_SAD(m_image[cur_row, right_col], m)

    beta = 1
    energy = e_spectral * math.exp(n_energy * beta)

    return energy


def infer_crf_datapoint(m_image, D_image, i, j, d):
    """
    Run metropolis algorithm (MCMC) to estimate m and D using CRF,
     - log(P(y_i | x_i)) + sum_{n in neighbors} SAD(y_i, y_n)
     Return m_image and D_image with updated values 
    :param iterations: Number of iterations to run over
    :param m_image: 3D Numpy array, mineral assemblages for pixels
    :param D_image: 3D Numpy array, grain sizes for pixels
    :param i: row index for datapoint d
    :param j: col index for datapoint d
    :param d: 1 spectral sample (1D Numpy vector)
    """
    cur_m = m_image[i, j]
    cur_D = D_image[i, j]
    # Determine whether or not to accept the new parameters, based on the
    # ratio of log (likelihood*priors)
    new_m, new_D = transition_model(cur_m, cur_D)

    cur = get_crf_posterior(m_image, D_image, i, j, cur_m, cur_D, d)
    new = get_crf_posterior(m_image, D_image, i, j, new_m, new_D, d)

    ratio = new / cur

    u = np.random.uniform(0, 1)
    if ratio > u:
        cur_m = new_m
        cur_D = new_D
    m_image[i, j] = cur_m
    D_image[i, j] = cur_D
    return m_image, D_image


def infer_crf_image(iterations, image):
    """
    Infer m and D for entire image by minimizing:
    - log(P(y_i | x_i)) + sum_{n in neighbors} SAD(y_i, y_n)
    using Gibbs sampling. 
    1. Initialize random mineral assemblages for each pixel
    2. Loop over pixels for X iteratinos, and use MCMC to sample new assemblage for each pixel. 
    :param iterations: Number of MCMC iterations to run for each datapoint 
    :param image: 3D Numpy array with 3rd dimension equal to len(c_wavelengths)
    """

    num_rows = image.shape[0]
    num_cols = image.shape[1]
    # Mineral assemblage predictions
    m_image = np.ones((num_rows, num_cols, 3))
    # Grain size predictions
    D_image = np.ones((num_rows, num_cols, 3))

    print("Initialize pixels in image... ")
    image_reflectances = image
    m_image, D_image = init_gibbs(image)
    # For X iterations
    for iteration in range(iterations):
        # Iterate over each pixel in image
        for i in range(num_rows):
            for j in range(num_cols):
                d = image[i, j]
                m_image, D_image = infer_crf_datapoint(m_image, D_image, i, j, d)
        print("Finished iteration " + str(iteration) + "/" + str(iterations))

    return m_image, D_image
