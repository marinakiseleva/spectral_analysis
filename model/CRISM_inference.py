"""
Inference using Hapke model, and same as other inference file except all methods leading up to inferring the likelihood are copied, with the new parameter 'angles' which is the incidence and emission angle per pixel.
"""

import numpy as np
from model.inference import *
from model.mrf_inference import *


def get_CRISM_likelihood(d, angles, m, D):
    """
    p(d|m, D)
    Get likelihood of reflectance spectra d, given the mineral assemblage and grain size.
    :param d: Spectral reflectance data, as Numpy vector
    :param m: Dict from SID to abundance
    :param D: Dict from SID to grain size
    """
    r_e = get_USGS_r_mixed_hapke_estimate(m, D, angles)
    covariance = np.zeros((len(d), len(d)))
    np.fill_diagonal(covariance,  0.01)  # 5 * (10 ** (-4)))

    y = multivariate_normal.pdf(x=d, mean=r_e, cov=covariance)

    # Threshold min values to not overflow
    if y < 10**-10:
        y = 10**-10

    return y


def get_CRISM_posterior_estimate(d, angles, m, D):
    """
    Get estimate of posterior
    p(m,D|d) = p(d|m, D) p(m) p(D)
    Which is just likelihood.

    Omit priors because they are uniform
    """
    m_dict = convert_arr_to_dict(m)
    D_dict = convert_arr_to_dict(D)
    l = get_CRISM_likelihood(d, angles, m_dict, D_dict)
    return l


def infer_CRISM_datapoint(d_seeds_index, iterations, V, C):
    """
    Run metropolis algorithm (MCMC) to estimate m and D. Return the MAP.
    :param d_seeds_index: [reflectance, angles, seed, pixel #] (1D Numpy vector)  
    :param V: covariance diagonal for grain size, D
    :param C: scaling factor for sampling mineral assemblage from Dirichlet, m
    """
    d, angles, seed, pixel_num = d_seeds_index
    np.random.seed(seed=seed)
    # Initialize randomly
    cur_m = sample_dirichlet(np.random.random(USGS_NUM_ENDMEMBERS), C)
    cur_D = np.full(shape=USGS_NUM_ENDMEMBERS, fill_value=INITIAL_D)
    unchanged_i = 0  # Number of iterations since last update
    MAP_mD = [cur_m, cur_D, 0]
    for i in range(iterations):
        new_m, new_D = transition_model(cur_m, cur_D, V, C)
        new_post = get_CRISM_posterior_estimate(d, angles, new_m, new_D)
        cur_post = get_CRISM_posterior_estimate(d, angles, cur_m, cur_D)

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


def infer_CRISM_points_parallel(iterations, V, C, d_seeds_indices):
    """
    Infer data points as pool
    """
    print("Starting parallel processing...")

    pool = multiprocessing.Pool(NUM_CPUS)

    # Pass in parameters that don't change for parallel processes
    func = partial(infer_CRISM_datapoint,
                   iterations=iterations,
                   V=V,
                   C=C)
    m_and_Ds = []
    m_and_Ds = pool.imap(func, d_seeds_indices)

    pool.close()
    pool.join()
    print("Done processing...")
    return m_and_Ds


def infer_CRISM_image(iterations, image, angle_img, V, C):
    """
    Same as infer_image but uses angle data for CRISM SSAs.
    Infer m and D for entire image - each pixel indepedently
    :param iterations: Number of MCMC iterations to run for each datapoint
    :param image: 3D Numpy array with 3d dimension equal to wavelengths
    :param angle_img: Image of same dimensions as image, but containing 2 indices, 0 - incidence angle, 1 - emission angle
    :param V: covariance diagonal for grain size, D
    :param C: scaling factor for sampling mineral assemblage from Dirichlet, m
    """

    angle_img = cos_img(angle_img)

    num_rows = image.shape[0]
    num_cols = image.shape[1]

    # index_coords Map from list index to X,Y coords in image
    index_coords = {}
    r_space = []  # List of reflectances for image
    angle_space = []
    index = 0
    for i in range(num_rows):
        for j in range(num_cols):
            index_coords[index] = [i, j]
            r_space.append(image[i, j])
            angle_space.append(angle_img[i, j])
            index += 1

    d_seeds_indices = []
    for i, r in enumerate(r_space):
        # Each parallel process needs random seed for sampling
        seed = np.random.randint(100000)
        d_seeds_indices.append([r_space[i],
                                angle_space[i],
                                seed,
                                i])

    m_and_Ds = infer_CRISM_points_parallel(iterations, V, C, d_seeds_indices)

    m_image, D_image = reconstruct_image(num_rows, num_cols, index_coords, m_and_Ds)

    return m_image, D_image


# MRF CRISM
def get_CRISM_mrf_prob(m_image, D_image, i, j, m, D, d, angles, beta):
    """
    Get joint probability of this pixel i,j in image
    """
    e_spatial = get_spatial_energy(m_image, i, j, m)

    p = get_CRISM_posterior_estimate(d, angles, m, D)

    # joint prob is likelihood - spatial energy
    return p - (e_spatial * beta)


def infer_CRISM_mrf_datapoint(m_image, D_image, i, j, d, angles, V, C, beta):
    """
    Run metropolis algorithm (MCMC) to estimate m and D using posterior
    Return m_image and D_image with updated values 
    :param m_image: 3D Numpy array, mineral assemblages for pixels
    :param D_image: 3D Numpy array, grain sizes for pixels
    :param i: row index for datapoint d
    :param j: col index for datapoint d
    :param d: data, 1 spectral sample (1D Numpy vector) 
    """
    cur_m = m_image[i, j]
    cur_D = D_image[i, j]
    new_m, new_D = transition_model(cur_m, cur_D, V, C)

    cur = get_CRISM_mrf_prob(m_image, D_image, i, j, cur_m, cur_D, d, angles, beta)
    new = get_CRISM_mrf_prob(m_image, D_image, i, j, new_m, new_D, d, angles, beta)

    ratio = new / cur
    phi = min(1, ratio)
    u = np.random.uniform(0, 1)
    if phi >= u:
        cur_m = new_m
        cur_D = new_D

    m_image[i, j] = cur_m
    D_image[i, j] = cur_D
    return m_image, D_image


def mrf_CRISM_iter(data, V, C, beta):
    """
    MCMC iteration for MRF model inference. Update m and D image.
    """
    m_image, D_image, r_image, angle_image, seed = data
    np.random.seed(seed=seed)

    rows = np.arange(0, m_image.shape[0])
    cols = np.arange(0, m_image.shape[1])

    # np.random.shuffle(cols)
    # np.random.shuffle(rows)
    # Iterate over each pixel in image
    for i in rows:
        for j in cols:
            d = r_image[i, j]
            angles = angle_image[i, j]
            m_image, D_image = infer_CRISM_mrf_datapoint(
                m_image, D_image, i, j, d, angles, V, C, beta)

    return m_image, D_image


def parallel_CRISM_mrf_iter(V, C, beta, m_image, D_image, r_image, angle_img):
    """
    MCMC iteration for MRF model inference. Update m and D image.
    Break image up into chunks and infer each in parallel.
    """
    num_rows = r_image.shape[0]
    num_cols = r_image.shape[1]
    rec_blocks = create_blocks(num_rows, num_cols)

    # For each block, save m, D, and reflectance for block of image
    img_data = []
    for block in rec_blocks:
        xmin, xmax, ymin, ymax = block
        m_S = m_image[xmin:xmax, ymin:ymax, :].copy()
        D_S = D_image[xmin:xmax, ymin:ymax, :].copy()
        r_S = r_image[xmin:xmax, ymin:ymax, :].copy()
        angle_S = angle_img[xmin:xmax, ymin:ymax, :].copy()
        seed = np.random.randint(100000)
        img_data.append([m_S, D_S, r_S, angle_S, seed])

    pool = multiprocessing.Pool(NUM_CPUS)

    func = partial(mrf_CRISM_iter, V=V, C=C, beta=beta)

    m_and_Ds = []
    m_and_Ds = pool.map(func, img_data)
    pool.close()
    pool.join()

    # Reconstruct from blocks
    for i, block in enumerate(rec_blocks):
        xmin, xmax, ymin, ymax = block

        m, D = m_and_Ds[i]
        m_image[xmin:xmax, ymin:ymax, :] = m
        D_image[xmin:xmax, ymin:ymax, :] = D

    return m_image, D_image


def infer_mrf_image_CRISM(beta, iterations, r_image, angle_img, V, C):
    """
    Infer m and D for entire image by minimizing:
    - log(P(y_i | x_i)) + sum_{n in neighbors} SAD(y_i, y_n) 
    1. Initialize random mineral assemblages for each pixel
    2. Loop over pixels for X iterations, and use MCMC to sample new assemblage for each pixel.
    :param beta: smoothness parameter 
    :param iterations: Number of MCMC iterations to run for each datapoint
    :param image: 3D Numpy array with 3rd dimension equal to # of wavelengths
    :param V: covariance diagonal for grain size, D
    :param C: scaling factor for sampling mineral assemblage from Dirichlet, m
    """
    print("Initialize MRF for beta=" + str(beta) + "... ")
    m_image, D_image = init_mrf(r_image, V, C)

    prev_energy = 0
    energy_diffs = []
    MAP_mD = [m_image, D_image, 1000]
    for iteration in range(iterations):

        start = time.time()

        m_image, D_image = parallel_CRISM_mrf_iter(
            V, C, beta, m_image, D_image, r_image, angle_img)

        # Print out iteration performance
        energy = get_total_energy(m_image, D_image, r_image, beta)
        # update MAP
        if energy < MAP_mD[2]:
            MAP_mD = [m_image, D_image, energy]

        energy_diff = energy - prev_energy
        energy_diffs.append(energy_diff)
        prev_energy = energy  # reset prev energy

        print_iter(iteration, iterations, beta, energy, energy_diff, start)

        if check_early_stop(energy_diffs, iteration):
            break

    return MAP_mD[:2]
