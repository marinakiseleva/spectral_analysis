"""
Generate fake image with USGS data and then evaluate it.
"""

import numpy as np
import math

from model.inference import infer_mrf_image
from preprocessing.generate_USGS_data import generate_image
from utils.plotting import plot_compare_highd_predictions
import utils.constants as consts


save_dir = "output/data/mrf/"


def run_mrf(image, mcmc_iterations):
    """
    Run MRF with passed-in distance metric
    """

    m_est, D_est = infer_mrf_image(iterations=mcmc_iterations,
                                   image=image)

    np.savetxt(save_dir + "m_estimated.txt", m_est.flatten())
    np.savetxt(save_dir + "D_estimated.txt", D_est.flatten())

    return m_est


def get_rmse(a, b):
    # RMSE
    return math.sqrt(np.mean((a - b)**2))

if __name__ == "__main__":
    num_mixtures = 5
    grid_res = 4
    noise_scale = 0.01  # 0.001
    res = 8
    mcmc_iterations = 50

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

    m_actual = image.m_image
    D_actual = image.D_image

    m_est = run_mrf(image.r_image, mcmc_iterations)
    m_rmse = str(round(get_rmse(m_actual, m_est), 2))

    print(" M estiamted ")
    print(m_est)

    # Save output
    np.savetxt(save_dir + "m_actual.txt", m_actual.flatten())
    np.savetxt(save_dir + "D_actual.txt", D_actual.flatten())
    p = plot_compare_highd_predictions(actual=m_actual,
                                       pred=m_est)

    # p.savefig("../output/figures/mrf/m_compare.png", bbox_inches='tight')
