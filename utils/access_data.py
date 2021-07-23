"""
Accesses RELAB and USGS data
"""
import pickle
import pandas as pd
import numpy as np
import spectral.io.envi as envi
from utils.constants import *
from preprocessing.USGS_preprocessing_helpers import *

####################################################
############# CRISM data access        #############
####################################################


def save_CRISM_RWs(new_USGS_W, new_CRISM_W):
    """
    Record reduced wavelengths
    """

    path = PREPROCESSED_DATA + "CRISM/"
    with open(path + 'RW_USGS.pickle', 'wb') as f:
        pickle.dump(new_USGS_W, f)
    with open(path + 'RW_CRISM.pickle', 'wb') as f:
        pickle.dump(new_CRISM_W, f)


def get_CRISM_RWs_USGS():
    """
    Get reduced wavelengths of USGS; those matched to CRISM
    """
    with open(PREPROCESSED_DATA + "CRISM/RW_USGS.pickle", 'rb') as handle:
        return pickle.load(handle)


def get_CRISM_RWs():
    """
    Get reduced wavelengths of CRISM; those matched to USGS
    """
    with open(PREPROCESSED_DATA + "CRISM/RW_CRISM.pickle", 'rb') as handle:
        return pickle.load(handle)


def save_CRISM_wavelengths(combined_ws):
    """
    Save wavelengths of matched CRISM img
    """
    with open(DATA_DIR + "PREPROCESSED_DATA/CRISM/" + 'CRISM_wavelengths.pickle', 'wb') as f:
        pickle.dump(combined_ws, f)


def get_CRISM_wavelengths(CRISM_match=False):
    """
    Get full set of wavelengths of CRISM data.
    """
    if CRISM_match:
        return get_CRISM_RWs()

    F = PREPROCESSED_DATA + "CRISM/CRISM_wavelengths.pickle"
    with open(F, 'rb') as handle:
        wavelengths = pickle.load(handle)
    return wavelengths


def save_CRISM_data(new_img, img_save_name):
    """
    Save CRISM img
    """
    with open(DATA_DIR + "PREPROCESSED_DATA/CRISM/" + img_save_name + '.pickle', 'wb') as f:
        pickle.dump(new_img, f)


def get_CRISM_data(file_name, img_dir):
    """
    Gets CRISM data and reduce wavelengths to match USGS.
    :param file_name: Full path and file name of Pickled CRISM image.
    :param img_dir: Directory of CRISM image.
    """
    with open(file_name, 'rb') as handle:
        loaded_img = pickle.load(handle)

    orig_wavelengths = get_CRISM_wavelengths(img_dir, True)
    matched_wavelengths = get_CRISM_RWs()

    indices = np.argwhere(np.isin(orig_wavelengths, matched_wavelengths)).flatten()
    return np.take(loaded_img, indices, axis=2)


####################################################
############# USGS data access         #############
####################################################
def clean_name(E):
    """
    Clean endmember name (to handle names like Olivine (Fo51))
    """
    for r in ["(", ")", " "]:
        E = E.replace(r, "")
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
    F = K_DIR + clean_name(endmember) + '.pickle'
    with open(F, 'wb') as handle:
        pickle.dump(data, handle)


def get_USGS_wavelengths(CRISM_match=False):
    if CRISM_match:
        return get_CRISM_RWs_USGS()

    F = PREPROCESSED_DATA + "USGS_wavelengths.pickle"
    with open(F, 'rb') as handle:
        return pickle.load(handle)


def get_USGS_preprocessed_data(endmember, CRISM_match=False):
    """
    Return reflectance for USGS endmember. 
    It has already been preprocessed.
    """
    r = None
    F = R_DIR + clean_name(endmember) + "_reflectance.pickle"
    with open(F, 'rb') as handle:
        r = pickle.load(handle)

    if CRISM_match:
        orig_wavelengths = get_USGS_wavelengths()
        matched_wavelengths = get_CRISM_RWs_USGS()
        indices = np.argwhere(np.isin(orig_wavelengths, matched_wavelengths)).flatten()
        r = np.take(r, indices)
    return r


####################################################
############# RELAB data access        #############
####################################################

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
