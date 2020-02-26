import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import numpy as np

from utils.constants import *
from utils.access_data import get_reflectance_spectra
from model.inference import get_log_likelihood
from model.hapke_model import get_r_mixed_hapke_estimate


def prep_file_name(text):
    """
    Remove unnecessary characters from text in order to save it as valid file name
    """
    replace_strs = ["\n", " ", ":", ".", ",", "/"]
    for r in replace_strs:
        text = text.replace(r, "_")
    return text


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

    # Set metadata names
    if SID not in sids_names:
        sid_name = "Mixture"
    else:
        sid_name = sids_names[SID]

    min_grain = min(D_map.values())
    max_grain = max(D_map.values())
    grain_text = ""
    if min_grain == max_grain:
        grain_text = str(min_grain)
    else:
        grain_text = str(min_grain) + " - " + str(max_grain)

    # Get reflectance
    data_reflectance = get_reflectance_spectra(SID, spectra_db)
    ll = get_log_likelihood(data_reflectance, m_map, D_map)
    # Derive reflectance from m and D
    estimated_reflectance = get_r_mixed_hapke_estimate(m_map, D_map)

    print("For " + sid_name)
    print("Log Likelihood: " + str(round(ll, 3)))
    print("SAD : " + str(get_SAD(data_reflectance, estimated_reflectance)))

    # fig, axes = plt.subplots(2, 1, constrained_layout=True)
    # fig, axes = plt.subplots(1, 2, dpi=200)
    fig, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    ax1.set_ylim((0, 1))
    ax1.plot(c_wavelengths, data_reflectance, label='Actual', color='crimson')
    ax1.set_title("Actual vs Estimated")
    ax1.set_ylabel("Reflectance")
    ax1.set_xlabel("Wavelength")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylim((0, 1))
    ax2.plot(c_wavelengths, estimated_reflectance, label='Estimated', color='pink')
    fig.legend()
    fig.suptitle(sid_name + ", for grain sizes: " + grain_text, fontsize=14, x=.6)
    # rect : tuple (left, bottom, right, top)
    fig.tight_layout(rect=[0, 0, 1.2, .9])


def plot_overlay_reflectances(SIDs, m_maps, D_maps, title):
    """
    Plots multiple reflectances on same plot
    """
    fig, ax = plt.subplots(1, 1, constrained_layout=True,
                           figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    colormap = cm.get_cmap('Paired')
    for index, m_map in enumerate(m_maps):
        D_map = D_maps[index]
        SID_name = sids_names[SIDs[index]]
        estimated_reflectance = get_r_mixed_hapke_estimate(m_map, D_map)

        min_grain = min(D_map.values())
        max_grain = max(D_map.values())
        grain_text = ""
        if min_grain == max_grain:
            grain_text = str(min_grain)
        else:
            grain_text = str(min_grain) + " - " + str(max_grain)

        ax.plot(c_wavelengths,
                estimated_reflectance,
                color=colormap.colors[index],
                label=SID_name + ", grain size : " + grain_text)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Reflectance")
        ax.legend()

        plt.ylim((0, 1))

    fig.suptitle(title, fontsize=14)

    fig.savefig("../output/data/" + prep_file_name(title))


def interpolate_image(img):
    """
    Convert 3d dimension of images to values between 0 and 1 (so they may be plotted as RGB)
    """
    return np.interp(img, (img.min(), img.max()), (0, 1))

from textwrap import wrap


def plot_compare_predictions(actual, preds, fig_title, subplot_titles, interp=False):
    """
    Compare actual to 2 different predictions
    :param actual: Numpy 3D array with 2 dimensions as image and 3d dimension as array of values (for actual image)
    :param preds: List of predicted images (2d Numpy arrays)
    :param fig_title: Title of entire Figure
    :param subplot_titles: Titles of sub-figures, 1 for each preds
    :param interp: if Numpy third dimension is not between 0 and 1 values, need to interpolate
    """
    if interp:
        actual = interpolate_image(actual)
        new_preds = []
        for pred in preds:
            new_preds.append(interpolate_image(pred))
        preds = new_preds

    num_subplots = len(preds) + 1

    fig, axes = plt.subplots(1, num_subplots, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    axes[0].imshow(actual)
    axes[0].set_title("Actual")
    for index, pred in enumerate(preds):
        axes[index + 1].imshow(pred)
        title = '\n'.join(wrap(subplot_titles[index], 14))
        axes[index + 1].set_title(title)

    fig.suptitle(t=fig_title,  fontsize=14)

    if num_subplots == 2:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=[0, 0.03, 1, 1.2])

    return fig
