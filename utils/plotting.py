import matplotlib.pyplot as plt
import numpy as np

from utils.constants import c_wavelengths, sids_names, NUM_ENDMEMBERS
from utils.access_data import get_reflectance_spectra
from model.inference import get_log_likelihood
from model.hapke_model import get_r_mixed_hapke_estimate


def plot_estimated_versus_actual(SID, spectra_db, m_map, D_map):
    """
    Compares passed-in sample spectra to derived spectra using m and D.
    """
    def get_SAD(a, b):
        """
        Get spectral angle distance:
            d(i,j) =  (i^T * j) / ( ||i|| ||j|| )
        :param a: Numpy vector
        :param b: Numpy vector
        """
        n = np.dot(a.transpose(), b)
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return np.arccos(n / d)

    data_reflectance = get_reflectance_spectra(SID, spectra_db)
    ll = get_log_likelihood(data_reflectance, m_map, D_map)
    print("For " + sids_names[SID])
    print("Log Likelihood: " + str(round(ll, 3)))

    estimated_reflectance = get_r_mixed_hapke_estimate(m_map, D_map)

    print("SAD : " + str(get_SAD(data_reflectance, estimated_reflectance)))

    fig, axes = plt.subplots(2, 1, constrained_layout=True)

    axes[0].plot(c_wavelengths, data_reflectance)
    axes[0].set_title("Actual")
    axes[0].set_ylabel("Reflectance")

    axes[1].plot(c_wavelengths, estimated_reflectance)
    axes[1].set_title("Estimated")
    axes[1].set_xlabel("Wavelength")
    axes[1].set_ylabel("Reflectance")
    fig.suptitle(sids_names[SID], fontsize=14)


def plot_compare(actual, pred, title, interp=False):
    """
    Plot images side by side
    :param actual: Numpy 3D array with 2 dimensions as image and 3d dimension as array of values (for actual image)
    :param pred: Numpy 3D array with 2 dimensions as image and 3d dimension as array of values (for predicted)
    :param interp: if Numpy third dimension is not between 0 and 1 values, need to interpolate
    """
    if interp:
        actual = np.interp(actual, (actual.min(), actual.max()), (0, 1))
        pred = np.interp(pred, (pred.min(), pred.max()), (0, 1))
    fig, axes = plt.subplots(1, 2, constrained_layout=True)

    if NUM_ENDMEMBERS != 3:
        raise ValueError(
            "Plotting with imshow only handles 3 endmembers- since this maps to R,G,B.")

    axes[0].imshow(actual)
    axes[0].set_title("Actual")

    axes[1].imshow(pred)
    axes[1].set_title("Estimated")
    fig.suptitle(title, fontsize=14)

    return fig
