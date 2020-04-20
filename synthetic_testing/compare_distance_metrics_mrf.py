"""
Main script to generate data and run inference on it.
"""

from model.segmentation import segment_image, get_superpixels
from model.inference import infer_mrf_image
from preprocessing.generate_data import generate_image
from utils.plotting import plot_compare_predictions
from utils.constants import NUM_ENDMEMBERS, DISTANCE_METRIC
import utils.constants as consts

import numpy as np
import math


save_dir = "../output/data/mrf/"


def run_mrf(distance_metric, image, mcmc_iterations):
    """
    Run MRF with passed-in distance metric
    """
    consts.DISTANCE_METRIC = distance_metric

    print("Reset distance metric to " + str(distance_metric))

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
    mcmc_iterations = 5

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

    m_est_SAD = run_mrf('SAD', image.r_image, mcmc_iterations)
    m_rmse_SAD = str(round(get_rmse(m_actual, m_est_SAD), 2))

    m_est_Euc = run_mrf('Euclidean', image.r_image, mcmc_iterations)
    m_rmse_Euc = str(round(get_rmse(m_actual, m_est_Euc), 2))

    # Save output
    np.savetxt(save_dir + "m_actual.txt", m_actual.flatten())
    np.savetxt(save_dir + "D_actual.txt", D_actual.flatten())
    p = plot_compare_predictions(actual=m_actual,
                                 preds=[m_est_SAD, m_est_Euc],
                                 fig_title="Mineral assemblage comparison",
                                 subplot_titles=[
                                     "MRF w/ SAD, RMSE: " + str(m_rmse_SAD), "MRF w/ Euclidean, RMSE: " + str(m_rmse_Euc)],
                                 interp=False)

    p.savefig("output/figures/mrf/m_compare.png", bbox_inches='tight')
