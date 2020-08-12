"""
Runs variational inference on the model to estimate the posterior p(m,D|d)
"""
import sys
from functools import partial
import multiprocessing
from datetime import datetime

from scipy.stats import multivariate_normal
from collections import OrderedDict
import numpy as np
import math
from model.hapke_model import get_USGS_r_mixed_hapke_estimate
import utils.constants as consts


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
    # variance which affects sampling rate
    SAMPLING_VARIANCE = 10

    length = mean.shape[0]
    covariance = np.zeros((length, length))
    np.fill_diagonal(covariance, SAMPLING_VARIANCE)

    sample = np.random.multivariate_normal(mean, covariance)

    # Ensure all max are <= 800 and all min >= 50
    for index, v in enumerate(sample):
        if v > 800:
            sample[index] = 800
        elif v < 50:
            sample[index] = 50
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
    return 1 / (consts.GRAIN_SIZE_MAX - consts.GRAIN_SIZE_MIN)


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
    # if y < 10**-310:
    #     y = 10**-310
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


def transition_model(cur_m, cur_D):
    """
    Sample new m and D
    :param cur_m: Vector of mineral abundances
    :param cur_D: Vector of grain sizes
    """
    new_m = sample_dirichlet(cur_m)
    new_D = sample_multivariate(cur_D)
    return new_m, new_D


def convert_USGS_arr_to_dict(values):
    """
    Converts Numpy array of proportions to dictionary from endmember name to proportion
    """
    d = OrderedDict()
    for index, v in enumerate(values):
        d[consts.USGS_PURE_ENDMEMBERS[index]] = v
    return d


def get_log_posterior_estimate(d, m, D):
    """
    Get estimate of posterior in log : log p(m, D|d)
    log p(d|m, D) + log p(m) + log p(D)
    """
    m_prior = math.log(get_m_prob(m))
    D_prior = math.log(get_D_prob(D))
    m_dict = convert_USGS_arr_to_dict(m)
    D_dict = convert_USGS_arr_to_dict(D)
    ll = get_log_likelihood(d, m_dict, D_dict)
    return ll + m_prior + D_prior


def get_posterior_estimate(d, m, D):
    """
    Get estimate of posterior
    p(m,D|d) = p(d|m, D) p(m) p(D)
    """
    m_prior = get_m_prob(m)
    D_prior = get_D_prob(D)
    m_dict = convert_arr_to_dict(m)
    D_dict = convert_arr_to_dict(D)
    ll = get_likelihood(d, m_dict, D_dict)
    return ll * m_prior * D_prior


def infer_datapoint(iterations, d):
    """
    Run metropolis algorithm (MCMC) to estimate m and D
    :param iterations: Number of iterations to run over
    :param d: 1 spectral sample (1D Numpy vector)
    """
    # Initialize with 1/# endmembers each mineral
    cur_m = np.array([float(1 / consts.USGS_NUM_ENDMEMBERS)]
                     * consts.USGS_NUM_ENDMEMBERS)
    cur_D = np.array([consts.INITIAL_D] * consts.USGS_NUM_ENDMEMBERS)

    unchanged_i = 0  # Number of iterations since last update
    groups = int(iterations / 10)
    hold_grain = True
    for g in range(groups):
        for i in range(iterations):

            # Determine whether or not to accept the new parameters, based on the
            # ratio of (likelihood*priors)
            new_m, new_D = transition_model(cur_m, cur_D)

            if hold_grain:
                new_D = cur_D
            else:
                new_m = cur_m

            cur_l = get_posterior_estimate(d, cur_m, cur_D)
            new_l = get_posterior_estimate(d, new_m, new_D)

            ratio = new_l / cur_l
            u = np.random.uniform(0, 1)
            if ratio > u:
                unchanged_i = 0
                cur_m = new_m
                cur_D = new_D
            else:
                unchanged_i += 1

            if unchanged_i > consts.EARLY_STOP:
                print("\nEarly Stop at " + str(i + (i * g)))
                break
        # Switch from holding m to D, or vice versa
        hold_grain = False if hold_grain == True else True

    print("Finished datapoint.")
    return [cur_m, cur_D]


def infer_segmented_image(iterations, superpixels):
    """
    Infer m and D for each superpixel independently
    :param superpixels: List of reflectances (each is Numpy array)
    """
    pool = multiprocessing.Pool(consts.NUM_CPUS)

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
    Infer m and D for entire image - each pixel indepedently
    :param iterations: Number of MCMC iterations to run for each datapoint
    :param image: 3D Numpy array with 3d dimension equal to wavelengths
    """

    num_rows = image.shape[0]
    num_cols = image.shape[1]

    # Mineral assemblage predictions
    m_image = np.ones((num_rows, num_cols, consts.USGS_NUM_ENDMEMBERS))
    # Grain size predictions
    D_image = np.ones((num_rows, num_cols, consts.USGS_NUM_ENDMEMBERS))

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

    pool = multiprocessing.Pool(consts.NUM_CPUS)

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
    :param image: 3D Numpy array with 3rd dimension equal to # of wavelengths
    """
    num_rows = image.shape[0]
    num_cols = image.shape[1]
    m_image = np.zeros((num_rows, num_cols, consts.USGS_NUM_ENDMEMBERS))
    D_image = np.zeros((num_rows, num_cols, consts.USGS_NUM_ENDMEMBERS))
    for i in range(num_rows):
        for j in range(num_cols):
            reflectance = image[i, j]

            rand_m = sample_dirichlet(
                np.array([float(1 / consts.USGS_NUM_ENDMEMBERS)] * consts.USGS_NUM_ENDMEMBERS))
            rand_D = sample_multivariate(
                np.array([consts.INITIAL_D] * consts.USGS_NUM_ENDMEMBERS))

            m_image[i, j] = rand_m
            D_image[i, j] = rand_D

    return m_image, D_image


def convert_arr_to_dict(values):
    """
    Convert Numpy array to dict
    """
    d = OrderedDict()
    for index, v in enumerate(values):
        d[consts.USGS_PURE_ENDMEMBERS[index]] = v
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

    if consts.DISTANCE_METRIC == 'SAD':
        # spectral angle distance, SAD
        return get_SAD(a, b)
    else:
        # Euclidean
        return np.linalg.norm(a - b)


def get_spatial_energy(m_image, i, j, m):
    """
    Get spatial energy using default distance
    :param m_image: 3D Numpy array, mineral assemblages for pixels
    :param i: row index for datapoint
    :param j: col index for datapoint
    :param m: Mineral assemblage for pixel i,j to consider
    """

    num_rows = m_image.shape[0]
    num_cols = m_image.shape[1]

    e_spatial = 0
    cur_row = i
    cur_col = j
    row_above = i - 1
    row_below = i + 1
    left_col = j - 1
    right_col = j + 1

    # Sum over distance between pixel and each neighbor

    if row_above >= 0:
        # Above
        e_spatial += get_distance(m_image[row_above, j], m)
        if left_col >= 0:
            # Top left
            e_spatial += get_distance(m_image[row_above, left_col], m)
        if right_col < num_cols:
            # Top right
            e_spatial += get_distance(m_image[row_above, right_col], m)

    if row_below < num_rows:
        # Below
        e_spatial += get_distance(m_image[row_below, j], m)
        if left_col >= 0:
            # Bottom left
            e_spatial += get_distance(m_image[row_below, left_col], m)
        if right_col < num_cols:
            # Top right
            e_spatial += get_distance(m_image[row_below, right_col], m)

    if left_col >= 0:
        # Left
        e_spatial += get_distance(m_image[cur_row, left_col], m)
    if right_col < num_cols:
        # Right
        e_spatial += get_distance(m_image[cur_row, right_col], m)

    return e_spatial


def get_mrf_energy(m_image, D_image, i, j, m, D, d):
    """
    Get energy at this pixel i,j in image
    """
    # get energy of neighbors
    e_spatial = get_spatial_energy(m_image, i, j, m)
    e_spectral = get_log_posterior_estimate(d, m, D)
    return -e_spectral + (e_spatial * consts.BETA)


def infer_mrf_datapoint(m_image, D_image, i, j, d):
    """
    Run metropolis algorithm (MCMC) to estimate m and D using posterior
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
    new_m, new_D = transition_model(cur_m, cur_D)

    cur = get_mrf_energy(m_image, D_image, i, j, cur_m, cur_D, d)
    new = get_mrf_energy(m_image, D_image, i, j, new_m, new_D, d)

    if new < cur:
        cur_m = new_m
        cur_D = new_D
    m_image[i, j] = cur_m
    D_image[i, j] = cur_D
    return m_image, D_image


def get_total_energy(image, m_image, D_image):
    """
    Get the total MRF energy; we want this to decrease at each iteration
    :param m_image: 3D Numpy array, mineral assemblages for pixels
    :param D_image: 3D Numpy array, grain sizes for pixels
    """
    num_rows = m_image.shape[0]
    num_cols = m_image.shape[1]

    energy_sum = 0
    for x in range(num_rows):
        for y in range(num_cols):
            d = image[x, y]
            m = m_image[x, y]
            D = D_image[x, y]
            e_spatial = get_spatial_energy(m_image, x, y, m)
            e_spectral = get_log_posterior_estimate(d, m, D)
            pixel_energy = -e_spectral + (e_spatial * consts.BETA)
            energy_sum += pixel_energy

    return energy_sum


def infer_mrf_image(iterations, image):
    """
    Infer m and D for entire image by minimizing:
    - log(P(y_i | x_i)) + sum_{n in neighbors} SAD(y_i, y_n)
    using Gibbs sampling.
    1. Initialize random mineral assemblages for each pixel
    2. Loop over pixels for X iteratinos, and use MCMC to sample new assemblage for each pixel.
    :param iterations: Number of MCMC iterations to run for each datapoint
    :param image: 3D Numpy array with 3rd dimension equal to # of wavelengths
    """
    num_rows = image.shape[0]
    num_cols = image.shape[1]
    # Mineral assemblage predictions
    m_image = np.ones((num_rows, num_cols, consts.USGS_NUM_ENDMEMBERS))
    # Grain size predictions
    D_image = np.ones((num_rows, num_cols, consts.USGS_NUM_ENDMEMBERS))

    print("Initialize pixels in image... ")
    image_reflectances = image
    m_image, D_image = init_gibbs(image)

    rows = np.arange(0, num_rows)
    cols = np.arange(0, num_cols)

    prev_energies = [0]
    energy_diffs = []
    for iteration in range(iterations):
        # Randomize order of rows and columns each iteration
        np.random.shuffle(cols)
        np.random.shuffle(rows)
        # Iterate over each pixel in image
        for i in rows:
            for j in cols:
                d = image[i, j]
                m_image, D_image = infer_mrf_datapoint(m_image, D_image, i, j, d)

        # Print out iteration performance
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        energy = get_total_energy(image, m_image, D_image)
        energy_diff = energy - prev_energies[-1]
        prev_energies.append(energy)
        energy_diffs.append(energy_diff)

        print("\n\n" + str(dt_string) + "  iteration " +
              str(iteration + 1) + "/" + str(iterations))
        print("Energy change: " + str(round(energy_diff, 4)))
        print("Total MRF Energy: " + str(round(energy, 4)))

        # ENERGY_CUTOFF = 5  # cutoff used for early stopping
        # if len(energy_diffs) > 25 and all(abs(i) <= ENERGY_CUTOFF
        #                                   for i in energy_diffs[-5:]):
        #     print("\nEARLY STOPPING AT THIS ITERATION.\n")
        #     break

    return m_image, D_image
