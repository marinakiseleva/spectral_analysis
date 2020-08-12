"""
Generate fake image with USGS data and then evaluate it with MRF model
"""
from datetime import datetime
import numpy as np
import math
import matplotlib.pyplot as plt
from spectral import *


from preprocessing.generate_USGS_data import generate_image

from utils.access_data import get_USGS_wavelengths
from utils.plotting import *
import utils.constants as consts

from model.hapke_model import get_USGS_r_mixed_hapke_estimate
from model.inference import infer_mrf_image, convert_arr_to_dict


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
    wavelengths = get_USGS_wavelengths(True)
    r_image = np.ones((num_rows, num_cols, len(wavelengths)))
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

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
    os.mkdir(consts.MODULE_DIR + "/output/" + dt_string)
    OUTPUT_DIR = consts.MODULE_DIR + "/output/" + dt_string + "/"

    num_mixtures = 5
    grid_res = 4
    noise_scale = 0.001  # 0.001
    res = 8
    mcmc_iterations = 1000

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
    np.savetxt(OUTPUT_DIR + "m_actual.txt", m_actual.flatten())
    np.savetxt(OUTPUT_DIR + "D_actual.txt", D_actual.flatten())
    with open(OUTPUT_DIR + "metadata.txt", "w") as text_file:
        text_file.write("MRF model with " + str(mcmc_iterations) +
                        " iterations on USGS test data")
    p = plot_compare_highd_predictions(actual=m_actual,
                                       pred=m_est,
                                       output_dir=OUTPUT_DIR)

    # Plot image in certain bands.
    bands = [80, 150, 220]
    actual_img = np.take(a=image.r_image[:, :], indices=bands, axis=2)
    est = estimate_image(m_est, D_est)
    estimated_img = np.take(a=est[:, :], indices=bands, axis=2)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    plot_as_rgb(actual_img, "Original", axes[0])
    plot_as_rgb(estimated_img, "Estimated", axes[1])
    fig.suptitle("Reflectance as RGB, using bands " +
                 str(bands) + "\n(m RMSE: " + str(m_rmse) + ")")
    fig.savefig(OUTPUT_DIR + "rgb_compare.png")
