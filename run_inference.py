"""
Main script to generate data and run inference on it. 
"""


from model.inference import infer_image
from preprocessing.generate_data import generate_image
from utils.plotting import plot_compare

import numpy as np
import math

if __name__ == "__main__":
    num_mixtures = 5
    grid_res = 4
    noise_scale = 0.001
    res = 4
    iterations = 1000

    # Print metadata
    print("Generating data with: ")
    print("\t" + str(num_mixtures) + " unique mixtures")
    print("\t" + str(noise_scale) + " noise (sigma)")
    print("\t" + str(grid_res) + " grid resolution")
    print("\t" + str(res) + " pixel resolution")
    print("\t" + str(iterations) + " iterations")

    print("Conducting MCMC with: ")
    print("\t" + str(iterations) + " iterations")

    image = generate_image(num_mixtures=num_mixtures,
                           grid_res=grid_res,
                           noise_scale=noise_scale,
                           res=res)

    m_est, D_est = infer_image(iterations=iterations,
                               image=image.r_image)
    m_actual = image.m_image
    D_actual = image.D_image

    # Save output
    save_dir = "output/data/independent/"
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
    fig_path = "output/figures/independent/"
    p = plot_compare(actual=m_actual,
                     pred=m_est,
                     title="Mineral assemblages m, as RGB (RMSE: " + m_rmse + ")")
    p.savefig(fig_path + "m_compare.png")
    p = plot_compare(actual=D_actual,
                     pred=D_est,
                     title="Grain size D, as RGB (RMSE: " + D_rmse + ")",
                     interp=True)
    p.savefig(fig_path + "D_compare.png")
