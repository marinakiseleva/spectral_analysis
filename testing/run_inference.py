"""
Run inference on synthetic data for any or all of the three methods.

"""
import numpy as np
import math
import os
import pickle

from model.models import *
from preprocessing.generate_USGS_data import generate_image
from utils.plotting import *
from utils.constants import *


def get_rmse(a, b):
    # Get RMSE error
    return math.sqrt(np.mean((a - b)**2))


def print_error(m_actual, D_actual, m_est, D_est):
    m_rmse = str(round(get_rmse(m_actual, m_est), 2))
    print("RMSE for m: " + m_rmse)
    D_rmse = str(round(get_rmse(D_actual, D_est), 2))
    print("RMSE for D: " + D_rmse)


def record_output(m_actual, D_actual, m_est, D_est, save_dir, exp_name):
    # Save output
    new_output_dir = "../output/" + exp_name + "/"
    if not os.path.exists(new_output_dir + "data/" + save_dir):
        os.makedirs(new_output_dir + "data/" + save_dir)
    if not os.path.exists(new_output_dir + "figures/" + save_dir):
        os.makedirs(new_output_dir + "figures/" + save_dir)

    data_save = new_output_dir + "data/" + save_dir 
    with open(data_save+"m_estimated.pickle", 'wb') as f:
        pickle.dump(m_est, f)
    with open(data_save+"D_estimated.pickle", 'wb') as f:
        pickle.dump(D_est, f)

    plot_highd_imgs(m_est, new_output_dir + "figures/" + save_dir, True, m_actual)
    plot_highd_imgs(D_est, new_output_dir + "figures/" + save_dir, False, D_actual)
    print_error(m_actual, D_actual, m_est, D_est)

    plot_compare_highd_predictions(
        actual=m_actual,
        pred=m_est,
        output_dir=new_output_dir + "figures/" + save_dir)

    # # Plot image in certain bands.
    # bands = [80, 150, 220]
    # actual_img = np.take(a=image.r_image[:, :], indices=bands, axis=2)
    # est = estimate_image(m_est, D_est)
    # estimated_img = np.take(a=est[:, :], indices=bands, axis=2)

    # fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    # plot_as_rgb(actual_img, "Original", axes[0])
    # plot_as_rgb(estimated_img, "Estimated", axes[1])
    # fig.suptitle("Reflectance as RGB, using bands " +
    #              str(bands) + "\n(m RMSE: " + str(m_rmse) + ")")


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


if __name__ == "__main__":
    num_mixtures = 5
    grid_res = 4
    noise_scale = 0.001
    res = 20
    iterations = 100

    EXP_NAME = "TEST_SYNTHETIC"

    if not os.path.exists('../output/' + EXP_NAME):
        os.makedirs('../output/' + EXP_NAME)

    with open(PREPROCESSED_DATA + "SYNTHETIC/m_actual.pickle", 'rb') as F:
        m_actual = pickle.load(F)
    with open(PREPROCESSED_DATA + "SYNTHETIC/D_actual.pickle", 'rb') as F:
        D_actual = pickle.load(F)
    with open(PREPROCESSED_DATA + "SYNTHETIC/r_img.pickle", 'rb') as F:
        R_image = pickle.load(F)

    m_actual = m_actual[18:20]
    D_actual = D_actual[18:20]
    R_image = R_image[18:20]
    
    # np.savetxt("../output/data/actual/m_actual.txt", m_actual.flatten())
    # np.savetxt("../output/data/actual/D_actual.txt", D_actual.flatten())

    m_est, D_est = ind_model(iterations=iterations,
                             image=R_image,
                             C=10,
                             V=50)
    print("Independent model error:")
    record_output(m_actual, D_actual, m_est, D_est, "ind/", EXP_NAME)

    plot_actual_m(m_actual, output_dir=MODULE_DIR + "/output/"+ EXP_NAME + "/")

    # m_est, D_est = seg_model(seg_iterations, iterations, image.r_image)
    # print("Seg model error:")
    # record_output(m_actual, D_actual, m_est, D_est, "seg/", EXP_NAME)

    # m_est, D_est = mrf_model(iterations=iterations,
    #                          image=image.r_image)
    # print("MRF model error:")
    # record_output(m_actual, D_actual, m_est, D_est, "mrf/", EXP_NAME)
