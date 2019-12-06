"""
Main script to generate data and run inference on it. 
"""

from model.segmentation import segment_image, get_superpixels
from model.inference import infer_segmented_image
from preprocessing.generate_data import generate_image
from utils.plotting import plot_compare

import numpy as np
import math

NUM_ENDMEMBERS = 3

if __name__ == "__main__":
    image = generate_image(num_mixtures=5,
                           num_regions=4,
                           noise_scale=0.001,
                           size=4)

    graphs = segment_image(iterations=1000,
                           image=image.r_image
                           )
    superpixels = get_superpixels(graphs)

    m_and_Ds = infer_segmented_image(iterations=1000,
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

    m_actual = image.m_image
    D_actual = image.D_image

    # Save output
    save_dir = "output/data/"
    np.savetxt(save_dir + "m_actual.txt", m_actual.flatten())
    np.savetxt(save_dir + "D_actual.txt", D_actual.flatten())
    np.savetxt(save_dir + "m_estimated.txt", m_est.flatten())
    np.savetxt(save_dir + "D_estimated.txt", D_est.flatten())

    # Print error
    def get_rmse(a, b):
        return math.sqrt(np.mean((a - b)**2))
    m_rmse = str(round(get_rmse(m_actual, m_est), 2))
    print("RMSE for m: " + m_rmse)
    D_rmse = str(round(get_rmse(D_actual, D_est), 2))
    print("RMSE for D: " + D_rmse)

    # Plot output
    p = plot_compare(actual=m_actual,
                     pred=m_est,
                     title="Mineral assemblages m, as RGB (RMSE: " + m_rmse + ")")
    p.savefig("output/figures/m_compare.png")
    p = plot_compare(actual=D_actual,
                     pred=D_est,
                     title="Grain size D, as RGB (RMSE: " + D_rmse + ")",
                     interp=True)
    p.savefig("output/figures/D_compare.png")
