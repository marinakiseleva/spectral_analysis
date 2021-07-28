"""
Estimate m and D per pixel of CRISM image. Image specified in constants file.  

NOTE, BEFORE RUNNING:
1. Define RW_USGS and RW_CRISM [CRISM Data Handling notebook]
2. Estimate k [preprocessing/estimatek.py]


Run 
python run_crism_inference.py

This will set up things to this specific CRISM file. It needs to be run each time the image is changed.


"""
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from spectral import *

from model.models import *
from model.hapke_model import get_USGS_r_mixed_hapke_estimate
from utils.plotting import *
from utils.access_data import *
import utils.constants as consts


def run_mrf(image, mcmc_iterations):
    """
    Run MRF with passed-in distance metric
    """

    m_est, D_est = infer_mrf_image(iterations=mcmc_iterations,
                                   image=image)
    np.savetxt(consts.MODULE_DIR + "/output/data/m_estimated.txt", m_est.flatten())
    np.savetxt(consts.MODULE_DIR + "/output/data/D_estimated.txt", D_est.flatten())
    return m_est, D_est


def save_data_and_figs(m_est, D_est, model_type):
    print("\nCompleted " + str(model_type) + " model.")
    # Save output


    data_save_dir = MODULE_DIR + "/output/data/crism/" + model_type + "/"
    fig_save_dir = MODULE_DIR + "/output/figures/crism/" + model_type + "/"

    #  Make all missing dirs.
    b = MODULE_DIR + "/output/"
    mkdirs = [b+"figures/", b+"data/", b+"figures/crism/", b+"data/crism/",
                b+"figures/crism/" + model_type +"/", 
                 b+"data/crism/" + model_type+ "/",  
                 data_save_dir, fig_save_dir]
    for m in mkdirs:
        if not os.path.exists(m):
            os.mkdir(m)

    np.savetxt(data_save_dir + "m_estimated.txt", m_est.flatten())
    np.savetxt(data_save_dir + "D_estimated.txt", D_est.flatten())

    plot_highd_imgs(img=m_est,
                    output_dir=fig_save_dir,
                    mOrD="m",
                    actual=None)
    plot_highd_imgs(img=D_est,
                    output_dir=fig_save_dir,
                    mOrD="D",
                    actual=None)

    # Compare reflectances in certain bands.
    wavelengths = get_CRISM_RWs()
    est = estimate_image_reflectance(m_est, D_est, wavelengths)
    # bands = [30, 80, 150]
    bands = [120, 71, 18]
    estimated_img = np.take(a=est[:, :], indices=bands, axis=2)
    plt.axis("off")
    plt.title("Bands: 120, 71, 18")
    plt.imshow(estimated_img, cmap='bone')
    plt.savefig("../output/figures/crism/" + model_type + "/est.pdf")

    return m_est, D_est


def estimate_image_reflectance(m, D, wavelengths):
    """
    Convert m and D estimates to reflectance, to visualize estimated image
    """
    num_wavelengths = len(wavelengths)
    print("Num of wavelengths here " + str(num_wavelengths))
    num_rows = m.shape[0]
    num_cols = m.shape[1]
    r_image = np.ones((num_rows, num_cols, num_wavelengths))
    for xindex, row in enumerate(range(num_rows)):
        for yindex, element in enumerate(range(num_cols)):
            cur_m = m[row, element]
            cur_D = D[row, element]
            m_dict = convert_arr_to_dict(cur_m)
            D_dict = convert_arr_to_dict(cur_D)
            restimate = get_USGS_r_mixed_hapke_estimate(m_dict, D_dict)
            r_image[row, element] = restimate
    return r_image


if __name__ == "__main__":
    iterations = 10
    seg_iterations = 30000

    # os.system("taskset -p -c 1-3 %d" % os.getpid())

    print("Conducting CRISM inference with: ")
    print("\t" + str(iterations) + " MCMC iterations")
    print("\t" + str(seg_iterations) + " segmentation iterations")

    F = PREPROCESSED_DATA + "CRISM/frt00010628.pickle"
    F2 = PREPROCESSED_DATA + "CRISM/frt00010628_angles.pickle"
    img = get_CRISM_data(F)
    with open(F2, 'rb') as handle:
        angle_img = pickle.load(handle)

    print("CRISM image size " + str(img.shape))

    img = img[5:10, 5:10]
    # Keep  0 and 1th indices, incidence and emission angles.
    angle_img = angle_img[5:10, 5:10, :2]

    # Independent
    m_est, D_est = ind_model(iterations=iterations,
                             image=img,
                             V=50,
                             C=10,
                             angle_img=angle_img)
    save_data_and_figs(m_est, D_est, "ind")

    # m_est, D_est = seg_model(seg_iterations, iterations, img)
    # save_data_and_figs(m_est, D_est, "seg")
    m_est, D_est = mrf_model(iterations=iterations,
                             image=img,
                             V=50,
                             C=10,
                             angle_img=angle_img)
    save_data_and_figs(m_est, D_est, "mrf")

    plot_colorbar()
