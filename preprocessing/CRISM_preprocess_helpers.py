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


# Helper functions

def remove_nulls(img):
    img = img[:, :, :]
    return np.where(img == 65535, 0, img)


def get_new_borders(img):
    """
    Get new row min/max and col min/max based on what pixel values are all zeros.
    """
    d = np.all(img == 65535, axis=2)
    row_min = 0
    row_max = img.shape[0]
    col_min = 0
    col_max = img.shape[1]
    for rowindex, row in enumerate(d):
        if row[0] == False:
            row_min = rowindex
            break

    for colindex, col in enumerate(d[0]):
        if col == False:
            col_min = colindex
            break

    i = row_max - 1
    while i >= 0:
        if row[i] == False:
            row_max = i + 1
            break
        i = i - 1

    j = col_max - 1
    while j >= 0:
        if d[0][j] == False:
            col_max = j + 1
            break
        j = j - 1

    print("Reduce image size to row range:" + str(row_min) + "-" + str(row_max) +
          " and col range: " + str(col_min) + "-" + str(col_max))
    print("Original range, row max= " + str(d.shape[0]) + ", col max=" + str(d.shape[1]))

    return row_min, row_max, col_min, col_max


def reduce_image(img, row_min, row_max, col_min, col_max):
    """
    Cut image down based on row min/max and col min/max
    Indexing is left-inclusive and right-exclusive. So, if we access data[:,:,10:13]
    we keep values at index 10, 11, and 12. 
    """
    return img[row_min:row_max, col_min:col_max, :]


def layer_CRISM(img_s, img_l,  pixel_dir, img_save_name):
    """
    Save layered image and corresponding layered wavelengths of 2 CRISM images, 
    in same directories they are from.
    """
    # Get wavelengths of each
    s_ws = pd.read_csv(pixel_dir + "spixel.csv",
                       header=None)[0].values
    l_ws = pd.read_csv(pixel_dir + "lpixel.csv",
                       header=None)[0].values

    new_img = layer_image(S_IMG=img_s,
                          L_IMG=img_l,
                          S_W=s_ws,
                          L_W=l_ws)

    # Get indices to drop (those that have any NULLs.)
    NULL_Rs = set()
    SAM_PIXEL = new_img[200, 200]
    for i, r in enumerate(SAM_PIXEL):
        if (r == 65535):
            print("drop i = " + str(i))
            NULL_Rs.add(i)

    NULL_Rs = list(NULL_Rs)
    print("Dropping indices: " + str(NULL_Rs))

    combined_ws = np.concatenate((s_ws, l_ws))

    # Make list of indices to keep based on those that we drop.
    keep_W_Indices = []
    for i in range(len(combined_ws)):
        if i not in NULL_Rs:
            keep_W_Indices.append(i)

    combined_ws = np.take(combined_ws, keep_W_Indices, axis=0)
    img = np.take(new_img, keep_W_Indices, axis=2)

    # Save joined image
    save_CRISM_data(img, img_save_name)

    # Save wavelengths used.
    save_CRISM_wavelengths(combined_ws)

    return img


def get_images(img_dir, d_img_name, s_img_name, l_img_name):
    """
    """
    # Replace all NULLs with zeros
    ddr_img = envi.open(file=img_dir + d_img_name + '.hdr')
    # ddr_img = remove_nulls(ddr_img)
    ddr_img = ddr_img[:, :, :]

    s_img = envi.open(file=img_dir + s_img_name + '.hdr')
    # s_img = remove_nulls(s_img)
    s_img = s_img[:, :, :]

    l_img = envi.open(file=img_dir + l_img_name + '.hdr')
    # l_img = remove_nulls(l_img)
    l_img = l_img[:, :, :]

    return ddr_img, s_img, l_img


# def record_layered_data(img_dir, pixel_dir, img_save_name):
#     """
#     Save layered image and corresponding layered wavelengths of 2 CRISM images,
#     in same directories they are from.
#     """
#     img_s = envi.open(file=img_dir + 's_trr3_CAT.hdr')
#     img_l = envi.open(file=img_dir + 'l_trr3_CAT.hdr')

#     # Get wavelengths of each
#     s_ws = pd.read_csv(pixel_dir + "spixel.csv",
#                        header=None)[0].values
#     l_ws = pd.read_csv(pixel_dir + "lpixel.csv",
#                        header=None)[0].values

#     new_img = layer_image(S_IMG=img_s[:, :, :],
#                           L_IMG=img_l[:, :, :],
#                           S_W=s_ws,
#                           L_W=l_ws)

#     # Replace NULL values with 0
#     new_img = np.where(new_img == 65535, 0, new_img)

#     # Save joined image
#     with open(DATA_DIR + "PREPROCESSED_DATA/" + img_save_name + '.pickle', 'wb') as f:
#         pickle.dump(new_img, f)

#     # Save wavelengths used.
#     combined_ws = np.concatenate((s_ws, l_ws))
#     with open(DATA_DIR + "PREPROCESSED_DATA/" + 'CRISM_wavelengths.pickle', 'wb') as f:
#         pickle.dump(combined_ws, f)


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


def record_CRISM_USGS_reduced_wavelengths():
    """
    Saves the wavelengths that are found to be as equal as possible 
    between USGS (olivine Fo80) and CRISM (random pixel from passed-in image) spectra. 
    """
    # Get data
    USGS_wavelengths = get_USGS_wavelengths()
    CRISM_wavelengths = get_CRISM_wavelengths()

    # Match USGS to CRISM
    precision = 0.002
    new_CRISM_W = []
    new_USGS_W = []
    for i, u in enumerate(USGS_wavelengths):
        for i_c, c in enumerate(CRISM_wavelengths):
            if abs(c - u) <= precision:
                new_CRISM_W.append(c)
                new_USGS_W.append(u)
                break

    save_CRISM_RWs(new_USGS_W, new_CRISM_W)

    rmse = np.sqrt(np.mean((np.array(new_CRISM_W) - np.array(new_USGS_W))**2))
    print("CRISM reduced from " + str(len(CRISM_wavelengths)) +
          " to " + str(len(new_CRISM_W)))
    print("USGS reduced from " + str(len(USGS_wavelengths)) +
          " to " + str(len(new_USGS_W)))
    print("RMSE between normalized wavelength vectors: " + str(rmse))

    return new_CRISM_W, new_USGS_W
