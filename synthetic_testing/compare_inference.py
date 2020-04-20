"""
Main script to generate data and run inference on it.
"""


from model.inference import infer_image, infer_segmented_image
from model.segmentation import segment_image, get_superpixels
from preprocessing.generate_data import generate_image
from utils.plotting import *
from utils.constants import NUM_ENDMEMBERS
import multiprocessing
import numpy as np
import math
import matplotlib.pyplot as plt


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


def run_independent_inference(iterations, image, rec=None):
    """
    Run pixel-independent inference
    """
    m_est, D_est = infer_image(iterations=iterations,
                               image=image.r_image)

    if rec is not None:
        rec['ind'] = [m_est, D_est]
    return m_est, D_est


def record_all_output(m_actual, D_actual, m_est_I, D_est_I, m_est_S, D_est_S):
    """
    """
    m_rmse_I, D_rmse_I = record_output(
        m_actual, D_actual, m_est_I, D_est_I, 'independent')
    m_rmse_S, D_rmse_S = record_output(m_actual, D_actual, m_est_S, D_est_S, 'segmented')

    m_titles = ["Independent (RMSE: " + m_rmse_I + ")",
                "Segmented (RMSE: " + m_rmse_S + ")"]
    p = plot_compare_predictions(actual=m_actual,
                                 preds=[m_est_I, m_est_S],
                                 fig_title="Mineral assemblage predictions as RGB",
                                 subplot_titles=m_titles)
    p.savefig("../output/figures/m_compare.png", bbox_inches='tight')

    D_titles = ["Independent (RMSE: " + D_rmse_I + ")",
                "Segmented (RMSE: " + D_rmse_S + ")"]
    p = plot_compare_predictions(actual=D_actual,
                                 preds=[D_est_I, D_est_S],
                                 fig_title="Grain size predictions as RGB",
                                 subplot_titles=D_titles,
                                 interp=True)

    p.savefig("../output/figures/D_compare.png", bbox_inches='tight')


def record_output(m_actual, D_actual, m_est, D_est, model_type):
    """
    Record error and image output
    :param model_type: independent or segmented
    """
    # Save output
    save_dir = "../output/data/" + model_type + "/"
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
    fig_path = "../output/figures/" + model_type + "/"
    p = plot_compare_predictions(actual=m_actual,
                                 preds=[m_est],
                                 fig_title="Mineral assemblage predictions as RGB",
                                 subplot_titles=[model_type + " (RMSE: " + m_rmse + ")"])
    p.savefig(fig_path + "m_compare.png")
    p = plot_compare_predictions(actual=D_actual,
                                 preds=[D_est],
                                 fig_title="Grain size predictions as RGB",
                                 subplot_titles=[model_type + " (RMSE: " + D_rmse + ")"],
                                 interp=True)

    p.savefig(fig_path + "D_compare.png")

    return m_rmse, D_rmse


if __name__ == "__main__":
    num_mixtures = 5
    grid_res = 4
    noise_scale = 0.01  # 0.001
    res = 8
    seg_iterations = 2  # 100000
    mcmc_iterations = 1  # 10000

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
    m_actual = image.m_image
    D_actual = image.D_image

    manager = multiprocessing.Manager()
    record = manager.dict()
    p1 = multiprocessing.Process(target=run_independent_inference,
                                 args=(mcmc_iterations, image, record))
    p1.start()
    p2 = multiprocessing.Process(target=run_segmented_inference,
                                 args=(seg_iterations, mcmc_iterations, image, record))
    p2.start()
    p1.join()
    p2.join()
    m_est_I, D_est_I = record['ind']
    m_est_S, D_est_S = record['seg']

    record_all_output(m_actual, D_actual, m_est_I, D_est_I, m_est_S, D_est_S)
