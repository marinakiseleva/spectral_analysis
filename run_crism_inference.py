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
from utils.access_data import *
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


def estimate_image_reflectance(m, D):
    """
    Convert m and D estimates to reflectance, to visualize estimated image
    """
    num_wavelengths = len(get_USGS_wavelengths(CRISM_match=True))

    num_rows = m.shape[0]
    num_cols = m.shape[1]
    r_image = np.ones((num_rows, num_cols, num_wavelengths))
    for xindex, row in enumerate(range(num_rows)):
        for yindex, element in enumerate(range(num_cols)):
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
    mcmc_iterations = 1

    print("Conducting MCMC with: ")
    print("\t" + str(mcmc_iterations) + " iterations")

    IMG_DIR = DATA_DIR + 'GALE_CRATER/cartOrder/cartorder/'
    image_file = IMG_DIR + 'layered_img_section.pickle'
    wavelengths_file = IMG_DIR + 'layered_wavelengths.pickle'

    # Normalize spectra across RELAB, USGS, and CRISM per each CRISM image
    # (since different CRISM images have different wavelengths)
    record_reduced_spectra(wavelengths_file)

    image = get_CRISM_data(image_file, wavelengths_file, CRISM_match=True)
    print("CRISM image size " + str(image.shape))

    m_est, D_est = run_mrf(image, mcmc_iterations)

    p = plot_highd_img(m_est)

    # Compare reflectances in certain bands.
    bands = [30, 80, 150]

    est = estimate_image_reflectance(m_est, D_est)
    estimated_img = np.take(a=est[:, :], indices=bands, axis=2)

    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

    plot_as_rgb(estimated_img, bands, "Estimated", ax)
    fig.suptitle("Reflectance as RGB, using bands " + str(bands))
    fig.savefig(consts.MODULE_DIR + "/output/figures/rgb.png")
