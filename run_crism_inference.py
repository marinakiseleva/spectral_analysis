"""
Estimate m and D per pixel of CRISM image. Image specified in constants file.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from spectral import *


from model.inference import infer_mrf_image, convert_arr_to_dict
from preprocessing.generate_USGS_data import generate_image
from model.hapke_model import get_USGS_r_mixed_hapke_estimate
from utils.plotting import *
import utils.constants as consts


def run_mrf(image, mcmc_iterations):
    """
    Run MRF with passed-in distance metric
    """

    m_est, D_est = infer_mrf_image(iterations=mcmc_iterations,
                                   image=image)
    np.savetxt(consts.MODULE_DIR + "/output/data/m_estimated.txt", m_est.flatten())
    np.savetxt(consts.MODULE_DIR + "/output/data/D_estimated.txt", D_est.flatten())
    return m_est, D_est


def estimate_image(m, D):
    """
    Convert m and D estimates to reflectance, to visualize estimated image
    """
    num_rows = m.shape[0]
    num_cols = m.shape[1]
    r_image = np.ones((num_rows, num_cols, consts.REDUCED_WAVELENGTH_COUNT))
    for row in range(num_rows):
        for element in range(num_cols):
            cur_m = m[row, element]
            cur_D = D[row, element]
            m_dict = convert_arr_to_dict(cur_m)
            D_dict = convert_arr_to_dict(cur_D)
            restimate = get_USGS_r_mixed_hapke_estimate(m_dict, D_dict)
            r_image[row, element] = restimate
    return r_image


def get_rmse(a, b):
    # RMSE
    return math.sqrt(np.mean((a - b)**2))

if __name__ == "__main__":
    num_mixtures = 5
    grid_res = 4
    noise_scale = 0.01  # 0.001
    res = 8
    mcmc_iterations = 1

    # Print metadata
    print("Generating data with: ")
    print("\t" + str(num_mixtures) + " unique mixtures")
    print("\t" + str(noise_scale) + " noise (sigma)")
    print("\t" + str(grid_res) + " grid resolution")
    print("\t" + str(res) + " pixel resolution")

    print("Conducting MCMC with: ")
    print("\t" + str(mcmc_iterations) + " iterations")

    image = get_CRISM_data()
    print("Image type " + str(type(image)))
    print("Image size " + str(image.shape))

    m_est, D_est = run_mrf(image, mcmc_iterations)

    p = plot_highd_img(m_est)

    # Compare reflectances in certain bands.
    bands = [30, 80, 150]
    est = estimate_image(m_est, D_est)
    estimated_img = np.take(a=est[:, :], indices=bands, axis=2)

    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

    plot_as_rgb(estimated_img, bands, "Estimated", ax)
    fig.suptitle("Reflectance as RGB, using bands " + str(bands))
    fig.savefig(consts.MODULE_DIR + "/output/figures/rgb.png")
