"""
Main script to generate data and run inference on it.
"""

from model.segmentation import segment_image, get_superpixels
from model.inference import infer_segmented_image
from preprocessing.generate_data import generate_image
from utils.constants import NUM_ENDMEMBERS

import numpy as np
import math


def run_segmented_inference(seg_iterations, mcmc_iterations, image, rec=None):
    """
    Run segmented inference on image
    """
    graphs = segment_image(iterations=seg_iterations,
                           image=image.r_image
                           )
    superpixels = get_superpixels(graphs)

    m_and_Ds = infer_segmented_image(iterations=mcmc_iterations,
                                     superpixels=superpixels)

    # Reconstruct image
    num_rows = image.r_image.shape[0]
    num_cols = image.r_image.shape[1]
    # Mineral assemblage predictions
    m_est = np.ones((num_rows, num_cols, NUM_ENDMEMBERS))
    # Grain size predictions
    D_est = np.ones((num_rows, num_cols, NUM_ENDMEMBERS))
    for index, pair in enumerate(m_and_Ds):
        graph = graphs[index]
        for v in graph.vertices:
            # retrieve x, y coords
            # [i, j] = index_coords[index]
            m, D = pair
            m_est[v.x, v.y] = m
            D_est[v.x, v.y] = D

    if rec is not None:
        rec['seg'] = [m_est, D_est]
    return m_est, D_est


if __name__ == "__main__":
    num_mixtures = 5
    grid_res = 4
    noise_scale = 0.01  # 0.001
    res = 8
    seg_iterations = 10
    mcmc_iterations = 5

    # Print metadata
    print("Generating data with: ")
    print("\t" + str(num_mixtures) + " unique mixtures")
    print("\t" + str(noise_scale) + " noise (sigma)")
    print("\t" + str(grid_res) + " grid resolution")
    print("\t" + str(res) + " pixel resolution")
    print("\t" + str(seg_iterations) + " iterations")

    print("Conducting MCMC with: ")
    print("\t" + str(mcmc_iterations) + " iterations")

    image = generate_image(num_mixtures=num_mixtures,
                           grid_res=grid_res,
                           noise_scale=noise_scale,
                           res=res)
    m_est, D_est = run_segmented_inference(
        seg_iterations, mcmc_iterations, image, rec=None)

    m_actual = image.m_image
    D_actual = image.D_image

    # Save output
    save_dir = "output/data/segmented/"
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
