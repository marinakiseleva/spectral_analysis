"""
Generate fake image with USGS data and then evaluate it.
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
    noise_scale = 0.001  # 0.001
    res = 8
    mcmc_iterations = 400

    # Print metadata
    print("Generating data with: ")
    print("\t" + str(num_mixtures) + " unique mixtures")
    print("\t" + str(noise_scale) + " noise (sigma)")
    print("\t" + str(grid_res) + " grid resolution")
    print("\t" + str(res) + " pixel resolution")

    print("Conducting MCMC with: ")
    print("\t" + str(mcmc_iterations) + " iterations")

    image = generate_image(num_mixtures=num_mixtures,
                           grid_res=grid_res,
                           noise_scale=noise_scale,
                           res=res)
    print("Image size " + str(image.r_image.shape))

    m_actual = image.m_image
    D_actual = image.D_image

    m_est, D_est = run_mrf(image.r_image, mcmc_iterations)
    m_rmse = str(round(get_rmse(m_actual, m_est), 4))
    D_rmse = str(round(get_rmse(D_actual, D_est), 4))
    print("Overall m RMSE : " + str(m_rmse))
    print("Overall D RMSE : " + str(D_rmse))

    # Save output
    np.savetxt(consts.MODULE_DIR + "/output/data/m_actual.txt", m_actual.flatten())
    np.savetxt(consts.MODULE_DIR + "/output/data/D_actual.txt", D_actual.flatten())
    p = plot_compare_highd_predictions(actual=m_actual,
                                       pred=m_est)

    # Compare reflectances in certain bands.
    bands = [30, 80, 150]
    actual_img = np.take(a=image.r_image[:, :], indices=bands, axis=2)
    est = estimate_image(m_est, D_est)
    estimated_img = np.take(a=est[:, :], indices=bands, axis=2)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

    plot_as_rgb(actual_img, bands, "Original", axes[0])
    plot_as_rgb(estimated_img, bands, "Estimated", axes[1])
    fig.suptitle("Reflectance as RGB, using bands " +
                 str(bands) + "\n(m RMSE: " + str(m_rmse) + ")")
    fig.savefig(consts.MODULE_DIR + "/output/figures/rgb_compare.png")
