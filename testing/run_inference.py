"""
Run inference on synthetic data for any or all of the three methods.

"""
import numpy as np
import math
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


def record_output(m_actual, D_actual, m_est, D_est, save_dir):
    # Save output
    np.savetxt("../output/data/" + save_dir + "m_estimated.txt", m_est.flatten())
    np.savetxt("../output/data/" + save_dir + "D_estimated.txt", D_est.flatten())

    plot_highd_imgs(m_est, "../output/figures/" + save_dir, True, m_actual)
    plot_highd_imgs(D_est, "../output/figures/" + save_dir, False, D_actual)
    print_error(m_actual, D_actual, m_est, D_est)

    plot_compare_highd_predictions(
        actual=m_actual,
        pred=m_est,
        output_dir=MODULE_DIR + "/output/figures/" + save_dir)

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
    iterations = 2500
    seg_iterations = 30000

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
    plot_actual_m(m_actual, output_dir=MODULE_DIR + "/output/figures/actual/")
    np.savetxt("../output/data/actual/m_actual.txt", m_actual.flatten())
    np.savetxt("../output/data/actual/D_actual.txt", D_actual.flatten())

    m_est, D_est = ind_model(iterations=iterations,
                             image=image.r_image)
    print("Independent model error:")
    record_output(m_actual, D_actual, m_est, D_est, "ind/")

    m_est, D_est = seg_model(seg_iterations, iterations, image.r_image)
    record_output(m_actual, D_actual, m_est, D_est, "seg/")

    m_est, D_est = mrf_model(iterations=iterations,
                             image=image.r_image)
    print("MRF model error:")
    record_output(m_actual, D_actual, m_est, D_est, "mrf/")
