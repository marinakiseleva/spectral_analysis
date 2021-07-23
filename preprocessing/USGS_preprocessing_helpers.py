"""
## Preprocess endmembers
Process endmembers. Remove values for which we are missing reflectances (first and last 9 values) from each endmember. Save wavelengths and reflectances as pickle
"""
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.constants import *


def get_orig_USGS_endmember(cur_endmember):
    ENDMEMBER_PATH = USGS_DATA + cur_endmember + "/"
    df = pd.read_csv(ENDMEMBER_PATH + "reflectance.txt",
                     delimiter="\t", names=['reflectance'], skiprows=1)
    df_W = pd.read_csv(ENDMEMBER_PATH + "wavelengths.txt",
                       delimiter="\t", names=['wavelength'], skiprows=1)
    wavelengths = df_W['wavelength'].values
    mags = df['reflectance'].values

    # replace each missing index with next non-missing index.
    missing_indices = np.where(mags < 0)[0]
    if len(missing_indices) > 0:
        print("Replacing " + str(len(missing_indices)) +
              "  missing/NULL mags with next available value to fill-in.")
        print(missing_indices)
    for missing_index in missing_indices:
        next_index = missing_index + 1
        while next_index is not None:
            if next_index >= len(mags):
                mags[missing_index] = 0
                break
            if mags[next_index] > 0:
                mags[missing_index] = mags[next_index]
                next_index = None
                break
            else:
                next_index = next_index + 1

    return wavelengths, mags


def preprocess_USGS():
    """
    Save reflectance for each USGS endmember in R_DIR. 
    Save USGS wavelengths (Same for each endmember) in PREPROCESSED_DATA. 
    """
    print("Saving the wavelengths in USGS endmembers.")
    endmembers = ["diopside", "augite", "pigeonite", "hypersthene",
                  "enstatite", "andesine", "labradorite", "olivineFo51", "magnetite"]

    #  Create R_DIR if it doesn't exist.

    if not os.path.exists(R_DIR):
        os.makedirs(R_DIR)

    for endmember in endmembers:
        W, M = get_orig_USGS_endmember(endmember)
        # Save endmember data as pickle, clipping first 9 and last 9 reflectances.
        W = W[10:-9]
        M = M[10:-9]
        with open(R_DIR + endmember + "_reflectance.pickle", 'wb') as f:
            pickle.dump(M, f)

        with open(PREPROCESSED_DATA + "USGS_wavelengths.pickle", 'wb') as f:
            pickle.dump(W, f)

    # WS=[]
    # for i, v in enumerate(WS):
    #     if i!=len(WS)-1:
    #         if (WS[i] == WS[i+1]).all():
    #             print(str(i) + " equals next")
    # print("All wavelengths equal each other.")


if __name__ == "__main__":
    preprocess_USGS()

    # fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    # ax.plot(wavelengths, mags,
    #             color=DARK_GREEN,
    #             label='olivine')
    # ax.set_ylabel("Reflectance")
    # ax.set_xlabel("Wavelength")
    # ax.set_ylim((0, 1))
    # ax.set_xlim((min(wavelengths), max(wavelengths)))
    # plt.legend()
    # plt.show()
