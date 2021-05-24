"""
Accesses RELAB and USGS data
"""
import pickle
import pandas as pd
import numpy as np
import spectral.io.envi as envi
from utils.constants import *


"""
CRISM data access
"""

def get_CRISM_wavelengths():
    """
    Get full set of wavelengths of CRISM data.
    """

    filename = DATA_DIR + "PREPROCESSED_DATA/" + 'CRISM_wavelengths.pickle'
    with open(filename, 'rb') as handle:
        wavelengths = pickle.load(handle)
    return wavelengths


def get_CRISM_RWs():
    """
    Get reduced wavelengths of CRISM; those matched to lab spectra
    """
    crism_w = DATA_DIR + "PREPROCESSED_DATA/FILE_CONSTANTS/RW_CRISM.pickle"
    with open(crism_w, 'rb') as handle:
        RW_CRISM = pickle.load(handle)
    return RW_CRISM


def get_CRISM_data(image_file):
    """
    Gets CRISM data this directory & name.
    :param image_file: File name of Pickled image.
    """
    if 'pickle' not in image_file:
        return ValueError("get_CRISM_data only handles pickles.")

    with open(image_file, 'rb') as handle:
        loaded_img = pickle.load(handle)

    path = DATA_DIR + "PREPROCESSED_DATA/"
    with open(path + "CRISM_wavelengths.pickle", 'rb') as handle:
        img_wavelengths = pickle.load(handle)

    RW_CRISM = get_CRISM_RWs()

    keep_indices = []  # indices of spectra to keep.
    for index, w in enumerate(img_wavelengths):
        if w in RW_CRISM:
            keep_indices.append(index)

    if len(keep_indices) != len(RW_CRISM):
        raise ValueError("Issue normalizing wavelengths of CRISM img. ")

    # Create new image with only these wavelengths
    newimg = np.zeros((loaded_img.shape[0], loaded_img.shape[1], len(RW_CRISM)))
    for xindex, row in enumerate(loaded_img):
        for yindex, val in enumerate(row):
            # Only keep values with reduced wavelengths
            newimg[xindex, yindex] = np.take(val, keep_indices)

    return newimg


"""
USGS data access
"""
def clean_name(E):
    """
    Clean endmember name (to handle names like Olivine (Fo51))
    """
    for r in ["(", ")", " "]:
        E = E.replace(r, "").lower()
    return E

def get_USGS_endmember_k(endmember):
    """
    Get k as numpy vector for endmember
    These were estimated using entire USGS spectra.
    """
    F = K_DIR + clean_name(endmember) + '.pickle'
    with open(F, 'rb') as handle:
        return np.array(pickle.load(handle)) 

def save_USGS_endmember_k(endmember, data):
    """
    Get k as numpy vector for endmember
    These were estimated using entire USGS spectra.
    """
    F = K_DIR + clean_name(endmember)  + '.pickle' 
    with open(F, 'wb') as handle:
        pickle.dump(data, handle)

def get_USGS_wavelengths(CRISM_match=False):
    if CRISM_match:
        raise ValueError("Do not handle CRISM-wavelength-matching for USGS yet.")
    F = PREPROCESSED_DATA  + "wavelengths.pickle"
    with open(F, 'rb') as handle:
        return pickle.load(handle) 

def get_USGS_preprocessed_data(endmember, CRISM_match=False):
    """
    Return reflectance for USGS endmember. 
    It has already been preprocessed.
    """
    if CRISM_match:
        raise ValueError("Do not handle CRISM-wavelength-matching for USGS yet.")

    F = R_DIR + clean_name(endmember) + "_reflectance.pickle" 
    with open(F, 'rb') as handle:
        return pickle.load(handle) 



# def get_USGS_data(endmember, CRISM_match=False):
#     """
#     Get USGS spectral reflectance data for this endmember, as Pandas DataFrame
#     If CRISM_match = True, then keep only reflectance values for wavelengths in RW_USGS.  
#     """
#     # Normalize filename
#     endmember = endmember.lower()
#     for r in ["(", ")", " "]:
#         endmember = endmember.replace(r, "")

#     # Open data in Pandas DataFrame
#     data = pd.read_csv(file_name)

#     # Replace NULL values (which are -1.23e34) with 0
#     data.loc[data['reflectance'] < 0, 'reflectance'] = 0

#     data = data.round(decimals=5)

#     if CRISM_match:
#         path = DATA_DIR + "PREPROCESSED_DATA/FILE_CONSTANTS/"
#         # Only keep rows with reduced wavelengths
#         with open(path + "RW_USGS.pickle", 'rb') as handle:
#             RW_USGS = pickle.load(handle)
#         rounded_RW_USGS = np.around(RW_USGS, decimals=5)
#         data = data[data['wavelength'].isin(rounded_RW_USGS)]

#     return data


"""
RELAB data access
"""


def get_data():
    """
    Get all RELAB data. Pull down and merge spectral data sources, return as Pandas DataFrame
    """

    file_name = CATALOGUE_PATH + "Minerals.xls"
    minerals = pd.read_excel(file_name)

    file_name = CATALOGUE_PATH + "Spectra_Catalogue.xls"
    Spectra_Catalogue = pd.read_excel(file_name)
    Spectra_Catalogue['SampleID'] = Spectra_Catalogue['SampleID'].str.strip()

    file_name = CATALOGUE_PATH + "Sample_Catalogue.xls"
    Sample_Catalogue = pd.read_excel(file_name)
    Sample_Catalogue['SampleID'] = Sample_Catalogue['SampleID'].str.strip()

    spectra_db = pd.merge(left=Spectra_Catalogue,
                          right=Sample_Catalogue,
                          on='SampleID')

    return spectra_db


def get_grain_sizes(spectrum_id, spectra_db):
    """
    Get range of grain sizes 
    :param spectrum_id: SpectrumID in dataset to look up
    :param spectra_db:  Merge of Spectra_Catalogue and Sample_Catalogue
    """
    s = spectra_db[spectra_db['SpectrumID'] == spectrum_id]
    min_grain_size = s['MinSize'].values[0]
    max_grain_size = s['MaxSize'].values[0]
    return min_grain_size, max_grain_size


def get_RELAB_wavelengths(spectrum_id, spectra_db, CRISM_match=False):
    r_data = get_reflectance_data(spectrum_id, spectra_db, CRISM_match)
    return r_data['Wavelength(micron)'].values


def get_reflectance_spectra(spectrum_id, spectra_db, CRISM_match=False):
    r_data = get_reflectance_data(spectrum_id, spectra_db, CRISM_match)
    return r_data['Reflectance'].values


def get_reflectance_data(spectrum_id, spectra_db, CRISM_match=False):
    """
    Returns spectral reflectance for the passed-in spectrum ID from RELAB
    :param spectrum_id: SpectrumID in dataset to look up
    :param spectra_db:  Merge of Spectra_Catalogue and Sample_Catalogue
    :param cut: Boolean on whether to keep only wavelenghts in c_wavelengths, or to use all.
    :return reflectance_df: Pandas DataFrame with 2 columns [Wavelength(micron), Reflectance]
    """
    pi = spectra_db[spectra_db['SpectrumID'] == spectrum_id]["PI"].values[0]
    sampleid = spectra_db[spectra_db['SpectrumID']
                          == spectrum_id]["SampleID"].values[0]

    pi = pi.lower()
    pre_sampleid = sampleid[0:2].lower()
    spectrum_id = spectrum_id.lower()
    file_name = RELAB_DATA_PATH + pi + "/" + pre_sampleid + "/" + spectrum_id + ".txt"

    reflectance_df = pd.read_csv(file_name, sep="\t", header=0, skiprows=1)

    if CRISM_match:
        # Only keep rows with reduced wavelengths
        with open(MODULE_DIR + "/utils/FILE_CONSTANTS/RW_BASALT.pickle", 'rb') as handle:
            RW_BASALT = pickle.load(handle)

        reflectance_df = reflectance_df.loc[
            reflectance_df['Wavelength(micron)'].isin(RW_BASALT)]

    return reflectance_df
