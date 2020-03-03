"""
Main script to generate data and run inference on it.
"""

from model.segmentation import segment_image, get_superpixels
from model.inference import infer_crf_image
from preprocessing.generate_data import generate_image
from utils.plotting import plot_compare_predictions
from utils.constants import NUM_ENDMEMBERS

import numpy as np
import math


if __name__ == "__main__":
    num_mixtures = 5
    grid_res = 4
    noise_scale = 0.01  # 0.001
    res = 8
    mcmc_iterations = 500

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

    m_est, D_est = infer_crf_image(iterations=mcmc_iterations,
                                   image=image.r_image)

    m_actual = image.m_image
    D_actual = image.D_image

    # Save output
    save_dir = "output/data/crf/"
    np.savetxt(save_dir + "m_actual.txt", m_actual.flatten())
    np.savetxt(save_dir + "D_actual.txt", D_actual.flatten())
    np.savetxt(save_dir + "m_estimated.txt", m_est.flatten())
    np.savetxt(save_dir + "D_estimated.txt", D_est.flatten())

    # Print error
    def get_rmse(a, b):
        return math.sqrt(np.mean((a - b)**2))
    m_rmse = str(round(get_rmse(m_actual, m_est), 2))
    D_rmse = str(round(get_rmse(D_actual, D_est), 2))
    print("RMSE for m: " + m_rmse)
    print("RMSE for D: " + D_rmse)

    p = plot_compare_predictions(actual=m_actual,
                                 preds=[m_est],  # add m_est_Original if want
                                 fig_title="Mineral assemblage comparison",
                                 subplot_titles=["CRF Estimated, RMSE: " + str(m_rmse)],
                                 interp=False)

    p.savefig("output/figures/crf/m_compare.png", bbox_inches='tight')
