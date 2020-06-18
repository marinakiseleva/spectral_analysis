import math
import numpy as np
import pandas as pd
from textwrap import wrap
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches


from utils.constants import *
from utils.access_data import *
from model.inference import get_log_likelihood
from model.hapke_model import get_synthetic_r_mixed_hapke_estimate


def prep_file_name(text):
    """
    Remove unnecessary characters from text in order to save it as valid file name
    """
    replace_strs = ["\n", " ", ":", ".", ",", "/"]
    for r in replace_strs:
        text = text.replace(r, "_")
    return text


def plot_as_rgb(img, bands, title, ax):
    """
    Plot 3d Numpy array as rgb image
    :param img: 3d numpy array
    :param bands: The 3 wavelength bands being used to visualize the data
    """
    ax.set_title(title)
    ax.imshow(img)


def plot_endmembers():
    """
    Plot wavelength vs reflectance for each endmember
    """
    names = ['olivine (Fo80)', 'olivine (Fo51)', 'augite',
             'labradorite', 'pigeonite', 'magnetite']
    endmembers = ['olivinefo80', 'olivinefo51', 'augite',
                  'labradorite', 'pigeonite', 'magnetite']
    colors = [LIGHT_GREEN, DARK_GREEN, LIGHT_BLUE, PINK, DARK_BLUE, RED]

    fig, ax = plt.subplots(figsize=(4, 4), dpi=140)

    for index, endmember in enumerate(endmembers):

        data = get_USGS_data(endmember, True)

        ax.plot(data['wavelength'],
                data['reflectance'],
                color=colors[index],
                label=names[index])

    # Plot RELAB basaltic glass
    ss = get_data()
    wavelengths = get_RELAB_wavelengths(
        spectrum_id='C1BE100', spectra_db=ss, CRISM_match=True)
    reflectance = get_reflectance_spectra(
        spectrum_id='C1BE100', spectra_db=ss, CRISM_match=True)
    ax.plot(wavelengths,
            reflectance,
            color='purple',
            label='basaltic glass')

    ax.set_ylim((0, 1))
    plt.legend()


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
    estimated_reflectance = get_synthetic_r_mixed_hapke_estimate(m_map, D_map)

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
        estimated_reflectance = get_synthetic_r_mixed_hapke_estimate(m_map, D_map)

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

    fig.savefig(MODULE_DIR + "/output/data/" + prep_file_name(title))


def interpolate_image(img):
    """
    Convert 3d dimension of images to values between 0 and 1 (so they may be plotted as RGB)
    """
    return np.interp(img, (img.min(), img.max()), (0, 1))


def plot_highd_img(img_m):
    """
    Plots each endmember on different heatmap plot
    :param img_m: Numpy 3D array with > 3 endmember proportions per pixel
    """
    for index, endmember in enumerate(USGS_PURE_ENDMEMBERS):
        fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
        endmember_img = img_m[:, :, index]
        axp = ax.imshow(endmember_img, vmin=0, vmax=1)
        cb = plt.colorbar(mappable=axp, ax=ax)
        ax.set_title(endmember)
        fig.savefig(MODULE_DIR + "/output/figures/m_" + endmember + ".png")

    return fig


def plot_compare_highd_predictions(actual, pred):
    """
    Compare actual to different predictions, for high-dimensional data (over 3 dimensions)
    :param actual: Numpy 3D array with > 3 endmember proportions per pixel
    :param pred: Predicted image 
    """

    for index, endmember in enumerate(USGS_PURE_ENDMEMBERS):
        fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
        endmember_actual = actual[:, :, index]
        endmember_pred = pred[:, :, index]

        axes[0].imshow(endmember_actual)
        axes[0].set_title("Actual")
        axp = axes[1].imshow(endmember_pred, vmin=0, vmax=1)

        rmse = math.sqrt(np.mean((endmember_actual - endmember_pred)**2))

        axes[1].set_title(endmember + " prediction\n" + "RMSE: " + str(round(rmse, 3)))

        cb = plt.colorbar(mappable=axp, ax=axes, location='right')

        fig.savefig(MODULE_DIR + "/output/figures/m_compare_" + endmember + ".png")

    return fig


def plot_compare_predictions(actual, preds, fig_title, subplot_titles, interp=False):
    """
    Compare actual to different predictions
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


def plot_zoomed_sectioned_CRISM(loaded_img, coords):
    """
    Plots original passed in image of frt0002037a_07_if165 and its zoomed in selected region used for testing.
    :param loaded_img: img 
    :param coords: List of [X, Y, max_x, max_y] where these values are based on the subsection of image: img[X:max_x, Y:max_y, :]

    """
    PLOTTING_BAND = 100  # 24
    print("plotting band " + str(PLOTTING_BAND))
    fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
    axp = ax.imshow(loaded_img[:, :, PLOTTING_BAND],
                    origin='upper', cmap='bone')

    # Rectangle params equal image section params as folows:
    # new_img[X:max_x, Y:max_y, :] ->
    # Rectangle((X, Y), max_x - X, max_y - Y, ... )
    # Rectangle ((bottom coord, left coord), width, height)
    if coords is None:
        raise ValueError("Coords must be passed in, and length 4.")
    else:
        X = coords[0]
        max_x = coords[1]
        Y = coords[2]
        max_y = coords[3]

    rect = patches.Rectangle((X, Y), max_x - X, max_y - Y, linewidth=1,
                             edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    # cb = plt.colorbar(mappable=axp, ax=ax)

    # axins = ax.inset_axes([0.4, 0.4, 0.47, 0.47])
    # axins.imshow(loaded_img[:, :, PLOTTING_BAND], vmin=0, vmax=.35, origin='upper')
    # # Top left and bottom right corners of box
    # x1, x2, y1, y2 = Y, max_y, X, max_x
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y2, y1)
    # axins.set_xticklabels('')
    # axins.set_yticklabels('')

    # bl_1 = [X, Y]  # Bottom left point on original
    # bl_2 = [300, 250]  # Bottom left point on zoom
    # tl_1 = [Y, Y + 200]  # Top left point on original
    # tl_2 = [Y, max_y]  # Top left point on zoom
    # plt.plot(bl_1, bl_2, tl_1, tl_2, color='red', linewidth=1)

    # Can't use because it flips inner image
    # ax.indicate_inset_zoom(axins, edgecolor='red')
    plt.savefig("frt0002037a_07_if165_zoomed")

    plt.title("frt0002037a_07_if165 CRISM Image")
    plt.show()
