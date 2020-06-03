"""
Functionality to layer CRISM images. Assumes CRISM images are TRDR of type 's' and 'l' where S = Visible-near infrared (0.4 - 1 µm) and L = Infrared (1 - 4 µm), and follow CRISM file naming conventions: http://crism.jhuapl.edu/data/CRISM_workshop_2017/Presentations/Ancillary/CRISM_File_Naming_Convention.pdf


"""
import pickle
import numpy as np
import spectral.io.envi as envi
from spectral import imshow

from utils.access_data import get_CRISM_wavelengths
from utils.constants import DATA_DIR
CRISM_path = DATA_DIR + 'GALE_CRATER/'


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


def record_layered_data(IMG_DIR, IMG_NAME, S_IMG_WAVELENGTHS, L_IMG_WAVELENGTHS):
    """
    Save layered image and corresponding layered wavelengths of 2 CRISM images, in same directories they are from
    """

    s_CRISM_file = CRISM_path + IMG_DIR + IMG_NAME + 's_trr3_CAT.img'
    img_s = envi.open(file=s_CRISM_file + '.hdr', image=s_CRISM_file)
    l_CRISM_file = CRISM_path + IMG_DIR + IMG_NAME + 'l_trr3_CAT.img'
    img_l = envi.open(file=l_CRISM_file + '.hdr', image=l_CRISM_file)

    # Get wavelengths of each
    s_wavelengths = get_CRISM_wavelengths(CRISM_path + S_IMG_WAVELENGTHS)
    l_wavelengths = get_CRISM_wavelengths(CRISM_path + L_IMG_WAVELENGTHS)

    new_img = layer_image(S_IMG=img_s[:, :, :],
                          L_IMG=img_l[:, :, :],
                          S_W=s_wavelengths,
                          L_W=l_wavelengths)

    # Replace NULL values with 0
    new_img = np.where(new_img == 65535, 0, new_img)

    # Save joined image
    with open(CRISM_path + IMG_DIR + 'layered_img.pickle', 'wb') as f:
        pickle.dump(new_img, f)

    # Save joined wavelengths
    s_wavelengths.sort()
    l_wavelengths.sort()
    ALL_W = np.concatenate((np.array(s_wavelengths), np.array(l_wavelengths)))
    with open(CRISM_path + IMG_DIR + 'layered_wavelengths.pickle', 'wb') as f:
        pickle.dump(ALL_W, f)
    return new_img


if __name__ == "__main__":
    # For image frt0002037a_07_if165
    IMG_DIR = 'cartOrder/cartorder/'
    IMG_NAME = 'frt0002037a_07_if165'
    S_IMG_WAVELENGTHS = IMG_DIR + 's_pixel_201_200.txt'
    L_IMG_WAVELENGTHS = IMG_DIR + 'l_pixel_x_201_y_200.txt'

    new_img = record_layered_data(
        IMG_DIR, IMG_NAME, S_IMG_WAVELENGTHS, L_IMG_WAVELENGTHS)

    # Also record section of image for testing
    with open(CRISM_path + IMG_DIR + 'layered_img_section.pickle', 'wb') as f:
        pickle.dump(new_img[10:50, 100:140, :], f)

    # For image frs00028346_01_if169
    IMG_DIR = 'cartOrder_part_0002/cartorder/'
    IMG_NAME = 'frs00028346_01_if169'
    S_IMG_WAVELENGTHS = IMG_DIR + 's_pixel_262_48.txt'
    L_IMG_WAVELENGTHS = IMG_DIR + 'l_pixel_201_91.txt'

    record_layered_data(IMG_DIR, IMG_NAME, S_IMG_WAVELENGTHS, L_IMG_WAVELENGTHS)

    # For viewing

    # with open(DATA_DIR + 'GALE_CRATER/' + IMG_DIR + 'layered_img.pickle', 'rb') as handle:
    #     loaded_img = pickle.load(handle)
    # view = imshow(loaded_img[10:420, 100:621, :], bands=(
    #     200, 71, 18), title="Layered " + IMG_NAME)

    # with open(DATA_DIR + 'GALE_CRATER/' + IMG_DIR + 'layered_img.pickle', 'rb') as handle:
    #     loaded_img = pickle.load(handle)
    # view = imshow(loaded_img[:182, 30:626, :], bands=(
    # 200, 71, 18), title="Layered " + IMG_NAME)
