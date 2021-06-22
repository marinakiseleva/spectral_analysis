"""
Runs  inference on the model to estimate the posterior p(m,D|d)
"""
import sys
from datetime import datetime
from functools import partial
import multiprocessing
import numpy as np
import math

from model.inference import *
from utils.constants import *



def init_mrf(r_image, V, C):
    """
    Set random mineral & grain  assemblage for each pixel and return 3D Numpy array with 3rd dimension as assemblage
    """
    N = USGS_NUM_ENDMEMBERS

    num_rows = r_image.shape[0]
    num_cols = r_image.shape[1]
    m_image = np.zeros((num_rows, num_cols, N))
    D_image = np.zeros((num_rows, num_cols, N))
    for i in range(num_rows):
        for j in range(num_cols):
            reflectance = r_image[i, j]

            rand_D = D_transition(np.array([INITIAL_D] * N), V)
            rand_m = sample_dirichlet(np.array([float(1 / N)] * N), C)
            m_image[i, j] = rand_m
            D_image[i, j] = rand_D

    return m_image, D_image


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

def get_total_energy(m_image, D_image, r_image, beta):
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
            d = r_image[x, y]
            m = m_image[x, y]
            D = D_image[x, y]
            e_spatial = get_spatial_energy(m_image, x, y, m)
            e_spectral = -get_log_posterior_estimate(d, m, D)
            pixel_energy = e_spectral + (e_spatial * beta)
            energy_sum += (pixel_energy)

    return energy_sum

def get_mrf_prob(m_image, D_image, i, j, m, D, d, beta):
    """
    Get joint probability of this pixel i,j in image
    """
    # get energy of neighbors
    e_spatial = get_spatial_energy(m_image, i, j, m)
    # Do not use log - posterior; for some reason that results in never
    # rejecting candidates.
    p = get_posterior_estimate(d, m, D)
    # joint prob is likelihood - spatial energy
    return p - (e_spatial * beta)


def infer_mrf_datapoint(m_image, D_image, i, j, d, V, C, beta):
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

    cur = get_mrf_prob(m_image, D_image, i, j, cur_m, cur_D, d, beta)
    new = get_mrf_prob(m_image, D_image, i, j, new_m, new_D, d, beta)

    ratio = new / cur
    phi = min(1, ratio)
    u = np.random.uniform(0, 1)
    if phi >= u:
        cur_m = new_m
        cur_D = new_D

    m_image[i, j] = cur_m
    D_image[i, j] = cur_D
    return m_image, D_image

def mrf_iter(data, V, C, beta):
    """
    MCMC iteration for MRF model inference. Update m and D image.
    """
    m_image, D_image, r_image, seed = data
    np.random.seed(seed=seed)

    rows = np.arange(0, m_image.shape[0])
    cols = np.arange(0, m_image.shape[1])

    # np.random.shuffle(cols)
    # np.random.shuffle(rows)
    # Iterate over each pixel in image
    for i in rows:
        for j in cols:
            d = r_image[i, j]
            m_image, D_image = infer_mrf_datapoint(
                m_image, D_image, i, j, d, V, C, beta)

    return m_image, D_image


def create_blocks(num_rows, num_cols):
    """
    Split image up into blocks over which to run parallel
    """
    # Given N CPUs, create N or more blocks where each block has a min width/height
    n_blocks = int(math.sqrt(NUM_CPUS))
    block_width = int(num_cols/n_blocks)
    block_height = int(num_rows/n_blocks)

    rec_blocks = []
    for i in range(n_blocks):
        for j in range(n_blocks):
            # x vals signify rows, y vals signify cols
            xmin = i*block_height
            xmax = xmin+block_height - 1
            
            ymin = j*block_width
            ymax = ymin+block_width - 1
            # If block is end of row or col, extend to fill rest of img
            if i == n_blocks - 1:
                xmax = num_rows - 1
            if j == n_blocks - 1:
                ymax = num_cols - 1
            # Need to +1 because end of numpy indexing is exclusive
            # i.e. x[10:12] gets values at indices 10 and 11, not 12.
            rec_blocks.append([xmin, xmax+1, ymin, ymax+1])
    return rec_blocks

def parallel_mrf_iter(V, C, beta, m_image, D_image, r_image):
    """
    MCMC iteration for MRF model inference. Update m and D image.
    Break image up into chunks and infer each in parallel.
    """
    num_rows = r_image.shape[0]
    num_cols = r_image.shape[1] 
    rec_blocks = create_blocks(num_rows, num_cols)
    

    # For each block, save m, D, and reflectance for block of image
    img_data =[]
    for block in rec_blocks:
        xmin, xmax, ymin, ymax = block
        m_S  = m_image[xmin:xmax, ymin:ymax, :].copy()
        D_S  = D_image[xmin:xmax, ymin:ymax, :].copy()
        r_S  = r_image[xmin:xmax, ymin:ymax, :].copy()
        seed = np.random.randint(100000)
        img_data.append([m_S, D_S, r_S, seed])

    pool = multiprocessing.Pool(NUM_CPUS)

    func = partial(mrf_iter, V=V, C=C, beta=beta)

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


def infer_mrf_image(beta, iterations, r_image, V, C):
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
    num_rows = r_image.shape[0]
    num_cols = r_image.shape[1] 
    m_image = np.ones((num_rows, num_cols, USGS_NUM_ENDMEMBERS)) 
    D_image = np.ones((num_rows, num_cols, USGS_NUM_ENDMEMBERS))

    print("Initialize pixels in image for beta=" + str(beta) + "... ") 
    m_image, D_image = init_mrf(r_image, V, C)

    rows = np.arange(0, num_rows)
    cols = np.arange(0, num_cols)

    prev_energy = 0
    prev_imgs = []  # save last MRF_PREV_STEPS imgs in case of early stopping
    energy_diffs = []
    MAP_mD = [m_image, D_image, 1000]
    for iteration in range(iterations):
        m_image, D_image = parallel_mrf_iter(V, C, beta, m_image, D_image, r_image)
        

        # Print out iteration performance
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        prev_imgs.append([m_image, D_image])
        energy = get_total_energy(m_image, D_image, r_image, beta)
        energy_diff = energy - prev_energy
        energy_diffs.append(energy_diff)

        # update MAP
        if energy < MAP_mD[2]:
            MAP_mD = [m_image, D_image, energy]

        prev_energy = energy  # reset prev energy
    
        ps = "Iteration " + str(iteration + 1) + "/" + str(iterations) 
        ps += " for beta=" + str(beta)
        ps += "; total MRF Energy: " + str(round(energy, 2))
        ps += "; energy change from last iteration (want negative): " + str(round(energy_diff, 2))
        print("\n"+ps)

        sys.stdout.flush()
        # If average energy change last MRF_PREV_STEPS runs was less than
        # MRF_EARLY_STOP, stop
        if len(energy_diffs) > MRF_BURN_IN:
            a_e = np.average(energy_diffs[-MRF_PREV_STEPS:])
            if a_e > MRF_EARLY_STOP:
                print("\nMRF Early Stop at iteration " +
                      str(iteration) + " with average energy " + str(a_e))
                m_image, D_image = prev_imgs[-MRF_PREV_STEPS]
                break


    return MAP_mD[:2] 
