"""
Layer CRISM images AND remove NULL values (65535)

Functionality to layer CRISM images. Assumes CRISM images are TRDR of type 's' and 'l' where S = Visible-near infrared (0.4 - 1 µm) and L = Infrared (1 - 4 µm), and follow CRISM file naming conventions: http://crism.jhuapl.edu/data/CRISM_workshop_2017/Presentations/Ancillary/CRISM_File_Naming_Convention.pdf


"""
import pickle
import numpy as np
import pandas as pd
import spectral.io.envi as envi
from spectral import imshow

from utils.access_data import *
from utils.constants import *


def layer_image(S_IMG, L_IMG, S_W, L_W):
    """
    Layer image by combining data at each pixel.
    :param S_IMG: Numpy array for s image 
    :param L_IMG: Numpy array for l image 
    :param S_W: wavelengths for s image
    :param L_W: wavelengths for l image
    """
    # Manually layer images
    num_rows_S = S_IMG.shape[0]
    num_cols_S = S_IMG.shape[1]
    num_rows_L = L_IMG.shape[0]
    num_cols_L = L_IMG.shape[1]
    if num_rows_S != num_rows_L or num_cols_S != num_cols_L:
        raise ValueError("l and s images not the same size. Can't be layered.")

    NEW_IMG = np.zeros((num_rows_S, num_cols_S, len(S_W) + len(L_W)))

    for x in range(num_rows_S):
        for y in range(num_cols_S):
            a = S_IMG[x, y]
            b = L_IMG[x, y]
            c = np.concatenate((np.array(a), np.array(b)))
            NEW_IMG[x, y] = c
    return NEW_IMG


def record_layered_data(img_dir, pixel_dir, img_save_name):
    """
    Save layered image and corresponding layered wavelengths of 2 CRISM images, in same directories they are from
    """
    img_s = envi.open(file=img_dir + 's_trr3_CAT.hdr')
    img_l = envi.open(file=img_dir + 'l_trr3_CAT.hdr')

    # Get wavelengths of each
    s_ws = pd.read_csv(pixel_dir + "spixel.csv",
                       header=None)[0].values
    l_ws = pd.read_csv(pixel_dir + "lpixel.csv",
                       header=None)[0].values

    new_img = layer_image(S_IMG=img_s[:, :, :],
                          L_IMG=img_l[:, :, :],
                          S_W=s_ws,
                          L_W=l_ws)

    # Replace NULL values with 0
    new_img = np.where(new_img == 65535, 0, new_img)

    # Save joined image
    with open(DATA_DIR + "PREPROCESSED_DATA/" + img_save_name + '.pickle', 'wb') as f:
        pickle.dump(new_img, f)

    # Save wavelengths used.
    combined_ws = np.concatenate((s_ws, l_ws))
    with open(DATA_DIR + "PREPROCESSED_DATA/" + 'CRISM_wavelengths.pickle', 'wb') as f:
        pickle.dump(combined_ws, f)


def match_lists(CRISM_wavelengths, USGS_wavelengths):
    """
    Keeps wavelengths that are same in 2 lists. Tested on wavelength vectors which contains same subset of values.   
    """
    new_CRISM = []
    new_USGS = []
    U_index = 0
    for index, C_val in enumerate(CRISM_wavelengths):
        if U_index == len(USGS_wavelengths):
            break

        U_val = USGS_wavelengths[U_index]

        if C_val >= U_val:
            U_index += 1
            new_CRISM.append(C_val)
            new_USGS.append(U_val)

    return new_CRISM, new_USGS


def record_reduced_spectra():
    """
    Saves the wavelengths that are found to be as equal as possible between USGS (olivine Fo80)  and CRISM (random pixel from passed-in image) spectra.  The unique wavelengths are saved per data source because this is how their spectral values will be accessed.
    """
    # Get data
    USGS_data = get_USGS_data("olivine (Fo80)", CRISM_match=False)
    USGS_wavelengths = USGS_data['wavelength'].values.tolist()
    CRISM_wavelengths = get_CRISM_wavelengths()

    # Match USGS to CRISM
    CRISM_reduced, USGS_reduced = match_lists(CRISM_wavelengths, USGS_wavelengths)

    path = DATA_DIR + "PREPROCESSED_DATA/FILE_CONSTANTS/"
    with open(path + 'RW_USGS.pickle', 'wb') as f:
        pickle.dump(USGS_reduced, f)
    with open(path + 'RW_CRISM.pickle', 'wb') as f:
        pickle.dump(CRISM_reduced, f)

    rmse = np.sqrt(np.mean((np.array(CRISM_reduced) - np.array(USGS_reduced))**2))
    print("CRISM reduced from " + str(len(CRISM_wavelengths)) +
          " to " + str(len(CRISM_reduced)))
    print("USGS reduced from " + str(len(USGS_wavelengths)) +
          " to " + str(len(USGS_reduced)))
    print("RMSE between normalized wavelength vectors: " + str(rmse))
