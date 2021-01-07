"""
Run inference on each pixel independently.
"""
import numpy as np
import math

from model.inference import *
from preprocessing.generate_USGS_data import generate_image
from model.segmentation import segment_image, get_superpixels
from utils.plotting import *
from utils.constants import *


def get_rmse(a, b):
    # Print error
    return math.sqrt(np.mean((a - b)**2))


def print_error(m_actual, D_actual, m_est, D_est):
    m_rmse = str(round(get_rmse(m_actual, m_est), 2))
    print("RMSE for m: " + m_rmse)
    D_rmse = str(round(get_rmse(D_actual, D_est), 2))
    print("RMSE for D: " + D_rmse)


def infer_seg_model(seg_iterations, iterations, image, m_actual, D_actual):
    """
    Use segmentation model to infer mineral assemblages and grain sizes of pixels in image
    """
    graphs = segment_image(iterations=seg_iterations,
                           image=image)
    superpixels = get_superpixels(graphs)
    print("Number of superpixels: " + str(len(superpixels)))

    m_and_Ds = infer_segmented_image(iterations=iterations,
                                     superpixels=superpixels)

    # Reconstruct image
    num_rows = image.shape[0]
    num_cols = image.shape[1]
    # Mineral assemblage predictions
    m_est = np.ones((num_rows, num_cols, USGS_NUM_ENDMEMBERS))
    # Grain size predictions
    D_est = np.ones((num_rows, num_cols, USGS_NUM_ENDMEMBERS))
    for index, pair in enumerate(m_and_Ds):
        graph = graphs[index]
        for v in graph.vertices:
            # retrieve x, y coords
            # [i, j] = index_coords[index]
            m, D = pair
            m_est[v.x, v.y] = m
            D_est[v.x, v.y] = D

    # Save output
    save_dir = "../output/data/seg/"
    np.savetxt(save_dir + "m_estimated.txt", m_est.flatten())
    np.savetxt(save_dir + "D_estimated.txt", D_est.flatten())
    print("Seg model error:")
    print_error(m_actual, D_actual, m_est, D_est)
    plot_compare_highd_predictions(
        actual=m_actual, pred=m_est, output_dir=MODULE_DIR + "/output/figures/seg/")
    return m_est, D_est


def infer_ind_model(iterations, image, m_actual, D_actual):
    """
    Use pixel-independent model to infer mineral assemblages and grain sizes of pixels in image
    """

    m_est, D_est = infer_image(iterations=iterations,
                               image=image.r_image)

    # Save output
    save_dir = "../output/data/ind/"
    np.savetxt(save_dir + "m_estimated.txt", m_est.flatten())
    np.savetxt(save_dir + "D_estimated.txt", D_est.flatten())
    print("Independent model error:")
    print_error(m_actual, D_actual, m_est, D_est)
    plot_compare_highd_predictions(
        actual=m_actual, pred=m_est, output_dir=MODULE_DIR + "/output/figures/ind/")
    return m_est, D_est


def infer_mrf_model(iterations, image, m_actual, D_actual):
    """
    Use pixel-independent model to infer mineral assemblages and grain sizes of pixels in image
    """
    m_est, D_est = infer_mrf_image(iterations=iterations,
                                   image=image)

    # Save output
    save_dir = "../output/data/mrf/"
    np.savetxt(save_dir + "m_estimated.txt", m_est.flatten())
    np.savetxt(save_dir + "D_estimated.txt", D_est.flatten())
    print("MRF model error:")
    print_error(m_actual, D_actual, m_est, D_est)
    plot_compare_highd_predictions(
        actual=m_actual, pred=m_est, output_dir=MODULE_DIR + "/output/figures/mrf/")
    return m_est, D_est


if __name__ == "__main__":
    num_mixtures = 5
    grid_res = 4
    noise_scale = 0.001
    res = 4
    iterations = 500
    seg_iterations = 3000

    # Print metadata
    print("Generating data with: ")
    print("\t" + str(num_mixtures) + " unique mixtures")
    print("\t" + str(noise_scale) + " noise (sigma)")
    print("\t" + str(grid_res) + " grid resolution")
    print("\t" + str(res) + " pixel resolution")
    print("\t" + str(iterations) + " iterations")
    image = generate_image(num_mixtures=num_mixtures,
                           grid_res=grid_res,
                           noise_scale=noise_scale,
                           res=res)
    m_actual = image.m_image
    D_actual = image.D_image
    np.savetxt("../output/data/actual/m_actual.txt", m_actual.flatten())
    np.savetxt("../output/data/actual/D_actual.txt", D_actual.flatten())

    # infer_ind_model(iterations, image.r_image, m_actual, D_actual)

    infer_seg_model(seg_iterations, iterations, image.r_image, m_actual, D_actual)

    # infer_mrf_model(iterations, image.r_image, m_actual, D_actual)
