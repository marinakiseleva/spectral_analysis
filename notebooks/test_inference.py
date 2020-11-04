import numpy as np
from spectral import imshow
import spectral.io.envi as envi

from utils.constants import *
from utils.plotting import *
from utils.access_data import get_USGS_wavelengths

from model.inference import *
from model.hapke_model import get_USGS_r_mixed_hapke_estimate

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)


# m_random = np.random.dirichlet(np.ones(7), size=1)[0]
# D_random = np.random.randint(low=45, high=100, size=7)

m_random = np.array([0, 0, 0, 0, 0, 0, 1])
D_random = np.array([60, 60, 60, 60, 60, 60, 60])


print(m_random)
print(D_random)


m = convert_arr_to_dict(m_random)
D = convert_arr_to_dict(D_random)

r_actual = get_USGS_r_mixed_hapke_estimate(m=m, D=D)


est_m, est_D = infer_datapoint(iterations=120, d=r_actual)


print("Estimated Mineral assemblages: ")
print(est_m)
print("Actual Mineral assemblages: ")
print(m_random)

print("Estimated Grain sizes:")
print(est_D)
print("Actual Grain Sizes: ")
print(D_random)


wavelengths = get_USGS_wavelengths(True)
r_est = get_USGS_r_mixed_hapke_estimate(convert_arr_to_dict(est_m),
                                        convert_arr_to_dict(est_D))
fig, ax = plt.subplots(1, 1, constrained_layout=True,
                       figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
ax.plot(wavelengths, r_est, label="Estimated")
ax.plot(wavelengths, r_actual, label="Actual")

ax.set_xlabel("Wavelength")
ax.set_ylabel("Reflectance")
ax.legend()

plt.ylim((0, 1))
plt.savefig("ref_estimate.png")
