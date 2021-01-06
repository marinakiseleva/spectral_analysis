"""
Main script to generate data and run inference on it.
Uses all USGS endmembers and basaltic glass from RELAB.
"""
import multiprocessing
import numpy as np
import math
import matplotlib.pyplot as plt


from model.inference import infer_mrf_image, infer_segmented_image, infer_image
from model.segmentation import segment_image, get_superpixels
from preprocessing.generate_USGS_data import generate_image
from utils.plotting import *
from utils.constants import *


def run_segmented_inference(seg_iterations, iterations, image, num_endmembers, rec=None):
    """
    Run segmented inference on image
    """
    graphs = segment_image(iterations=seg_iterations,
                           image=image.r_image
                           )
    superpixels = get_superpixels(graphs)
    print("Number of superpixels: " + str(len(superpixels)))

    m_and_Ds = infer_segmented_image(iterations=iterations,
                                     superpixels=superpixels)

    # Reconstruct image
    num_rows = image.r_image.shape[0]
    num_cols = image.r_image.shape[1]
    # Mineral assemblage predictions
    m_est = np.ones((num_rows, num_cols, num_endmembers))
    # Grain size predictions
    D_est = np.ones((num_rows, num_cols, num_endmembers))
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


def run_mrf_inference(iterations, image, rec=None):
    """
    Run pixel-independent inference
    """
    m_est, D_est = infer_mrf_image(iterations=iterations,
                                   image=image.r_image)

    if rec is not None:
        rec['mrf'] = [m_est, D_est]
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
        m_actual, D_actual, m_est_I, D_est_I, 'mrf')
    m_rmse_S, D_rmse_S = record_output(m_actual, D_actual, m_est_S, D_est_S, 'segmented')

    m_titles = ["MRF (RMSE: " + m_rmse_I + ")",
                "Segmented (RMSE: " + m_rmse_S + ")"]
    p = plot_compare_predictions(actual=m_actual,
                                 preds=[m_est_I, m_est_S],
                                 fig_title="Mineral assemblage predictions as RGB",
                                 subplot_titles=m_titles)
    p.savefig("output/figures/m_compare.png", bbox_inches='tight')

    D_titles = ["MRF (RMSE: " + D_rmse_I + ")",
                "Segmented (RMSE: " + D_rmse_S + ")"]
    p = plot_compare_predictions(actual=D_actual,
                                 preds=[D_est_I, D_est_S],
                                 fig_title="Grain size predictions as RGB",
                                 subplot_titles=D_titles,
                                 interp=True)

    p.savefig("output/figures/D_compare.png", bbox_inches='tight')


def get_rmse(a, b):
    return math.sqrt(np.mean((a - b)**2))


def record_output(m_actual, D_actual, m_est, D_est, model_type):
    """
    Record error and image output
    :param model_type: independent or segmented
    """
    # Save output
    save_dir = "output/data/" + model_type + "/"
    np.savetxt(save_dir + "m_actual.txt", m_actual.flatten())
    np.savetxt(save_dir + "D_actual.txt", D_actual.flatten())
    np.savetxt(save_dir + "m_estimated.txt", m_est.flatten())
    np.savetxt(save_dir + "D_estimated.txt", D_est.flatten())

    # Print error

    m_rmse = str(round(get_rmse(m_actual, m_est), 2))
    print("RMSE for m: " + m_rmse)
    D_rmse = str(round(get_rmse(D_actual, D_est), 2))
    print("RMSE for D: " + D_rmse)

    # Plot output
    fig_path = "output/figures/" + model_type + "/"
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
    noise_scale = 0.001  # 0.001
    res = 8
    seg_iterations = 10000
    mcmc_iterations = 10  # 10000

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
    p1 = multiprocessing.Process(target=run_mrf_inference,
                                 args=(mcmc_iterations, image, record))
    p1.start()
    p2 = multiprocessing.Process(target=run_segmented_inference,
                                 args=(seg_iterations, 300, image, USGS_NUM_ENDMEMBERS, record))
    p2.start()

    p3 = multiprocessing.Process(target=run_independent_inference,
                                 args=(10, image, record))
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    m_est_C, D_est_C = record['mrf']
    m_est_S, D_est_S = record['seg']
    m_est_I, D_est_I = record['ind']

    C_rmse = get_rmse(m_est_C, m_actual)
    S_rmse = get_rmse(m_est_S, m_actual)
    I_rmse = get_rmse(m_est_I, m_actual)
    print("Shape of m actual " + str(m_actual.shape))
    print("Shape of m estimate for MRF " + str(m_est_C.shape))
    print("Shape of m estimate for seg " + str(m_est_S.shape))
    print("Shape of m estimate for ind " + str(m_est_I.shape))

    m_titles = ["MRF (RMSE: " + str(round(C_rmse, 2)) + ")",
                "Segmented (RMSE: " + str(round(S_rmse, 2)) + ")",
                "Independent (RMSE: " + str(round(I_rmse, 2)) + ")"]

    plot_compare_highd_predictions(actual=m_actual, pred=m_est_C,
                                   output_dir=MODULE_DIR + "/output/figures/mrf/")

    plot_compare_highd_predictions(actual=m_actual, pred=m_est_S,
                                   output_dir=MODULE_DIR + "/output/figures/seg/")

    plot_compare_highd_predictions(actual=m_actual, pred=m_est_I,
                                   output_dir=MODULE_DIR + "/output/figures/ind/")
