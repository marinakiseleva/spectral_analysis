"""
Main script to generate data and run inference on it.
"""

from model.segmentation import segment_image, get_superpixels
from model.inference import infer_crf_image
from utils.plotting import plot_compare_predictions

import pickle
import numpy as np


def get_image(name):
    """
    Load image
    """
    data_dir = "/Users/marina/Documents/PhD/research/mars_research/data/custom/"
    with open(data_dir + name, 'rb') as f:
        x = pickle.load(f)
        return x

if __name__ == "__main__":
    mcmc_iterations = 5
    data_name = 'frt000047a3_07_if166l_trr3_CLEAN.pkl'

    # Print metadata
    print("Using CRISM data")
    print("Conducting MCMC with: ")
    print("\t" + str(mcmc_iterations) + " iterations")

    image = get_image(data_name)

    # Only go up to 211th band.
    image = image[50:60, 10:20, :211]

    m_est, D_est = infer_crf_image(iterations=mcmc_iterations,
                                   image=image)

    # Save output
    save_dir = "output/data/crf/"
    np.savetxt(save_dir + "m_estimated.txt", m_est.flatten())
    np.savetxt(save_dir + "D_estimated.txt", D_est.flatten())

    # Select 3 bands of CRISM image to visualize
    band1 = 50
    band2 = 120
    band3 = 200
    rgb_img = np.dstack((image[:, :, band1], image[:, :, band2], image[:, :, band3]))

    p = plot_compare_predictions(actual=rgb_img,
                                 preds=[m_est],  # add m_est_Original if want
                                 fig_title="Mineral assemblage comparison",
                                 subplot_titles=["CRF Estimated"],
                                 interp=False)

    p.savefig("output/figures/crf/CRISM_compare.png", bbox_inches='tight')
