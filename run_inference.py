from inference import infer_image
from generate_data import generate_image
from plotting import plot_compare

import numpy as np
import math

image = generate_image(num_mixtures=5,
                       num_regions=5,
                       noise_scale=0.001,
                       size=5)


m_est, D_est = infer_image(iterations=1000,
                           image=image.r_image)
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
