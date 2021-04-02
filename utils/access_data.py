"""
Accesses RELAB and USGS data
"""
import pickle
import pandas as pd
import numpy as np
import spectral.io.envi as envi
from utils.constants import *

"""
Helper functions to normalize the different wavelengths from the different data sources
"""


def get_endmember_wavelengths(CRISM_match=True):
    """
    Get matched wavelengths (USGS/RELAB/CRISM-matched) ; originally from RW_USGS
    """
    if CRISM_match == False:
        return ValueError("Only hanlde CRISM matching.")
    return N_WAVELENGTHS


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


def record_reduced_spectra(CRISM_w_file):
    """
    Saves the wavelengths that are found to be as equal as possible between USGS (olivine Fo80)  and CRISM (random pixel from passed-in image) spectra.  The unique wavelengths are saved per data source because this is how their spectral values will be accessed.
    :param CRISM_w_file: Path and .txt file name of CRISM z-profile for a single pixel, saved from CAT ENVI
    """
    # Get data
    ss = get_data()
    USGS_data = get_USGS_data("olivine (Fo80)", CRISM_match=False)
    USGS_wavelengths = USGS_data['wavelength'].values.tolist()
    CRISM_wavelengths = get_CRISM_wavelengths(CRISM_w_file)

    # Match USGS to CRISM
    CRISM_reduced, USGS_reduced = match_lists(CRISM_wavelengths, USGS_wavelengths)

    path = MODULE_DIR + "/utils/FILE_CONSTANTS/"
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


"""
CRISM data access
"""


def get_CRISM_img_wavelengths(wavelengths_file):
    """
    Get wavelengths that we are using in the CRISM image 
    """
    with open(wavelengths_file, 'rb') as handle:
        img_wavelengths = pickle.load(handle)

    crism_w = MODULE_DIR + "/utils/FILE_CONSTANTS/RW_CRISM.pickle"
    with open(crism_w, 'rb') as handle:
        # CRISM reduced wavelengths to keep.
        RW_CRISM = pickle.load(handle)

    keep_indices = []  # indices of spectra to keep.
    for index, w in enumerate(img_wavelengths):
        if w in RW_CRISM:
            keep_indices.append(index)

    if len(keep_indices) != len(RW_CRISM):
        raise ValueError("Issue normalizing wavelengths of CRISM img. ")

    return [img_wavelengths[i] for i in keep_indices]


def get_CRISM_data(image_file, wavelengths_file, CRISM_match=True):
    """
    Gets CRISM data this directory & name.
    If matching to CRISM - make sure that record_reduced_spectra was called before this so that we know what wavelengths to match to.
    :param image_file: Full dir and file name of Pickled image.
    :param wavelengths_file: Full dir and file name of wavelength values for image
    :param CRISM_match: filter spectra to same range as endmembers
    """
    if 'pickle' not in image_file:
        return ValueError("get_CRISM_data only handles pickles.")

    with open(image_file, 'rb') as handle:
        loaded_img = pickle.load(handle)

    if CRISM_match:
        """
        Keep all wavelengths in image that are in RW_CRISM
        """
        with open(wavelengths_file, 'rb') as handle:
            img_wavelengths = pickle.load(handle)

        crism_w = MODULE_DIR + "/utils/FILE_CONSTANTS/RW_CRISM.pickle"
        with open(crism_w, 'rb') as handle:
            # CRISM reduced wavelengths to keep.
            RW_CRISM = pickle.load(handle)

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

    return loaded_img


def get_CRISM_wavelengths(file):
    """
    Get the wavelengths of CRISM data from saved file. Handles pickle and CSV.
    :param file: Full file name of CRISM wavelengths (including path)
    """
    if file is None:
        return ValueError("Must pass in file.")

    if ".csv" in file:
        data = pd.read_csv(file, header=None)
        wavelengths = data[0]
    elif '.pickle' in file:
        with open(file, 'rb') as handle:
            wavelengths = pickle.load(handle)
    return wavelengths


"""
USGS data access
"""


def get_USGS_wavelengths(CRISM_match=False):
    """
    Get wavelengths for the endmember as numpy vector
    """
    # Default to olivine (Fo51) wavelengths
    data = get_USGS_data(endmember='olivine (Fo51)', CRISM_match=CRISM_match)
    return data['wavelength'].values


def get_USGS_endmember_k(endmember):
    """
    Get k as numpy vector for endmember
    These were estimated using entire USGS spectra.
    """
    file_name = ENDMEMBERS_K + endmember

    with open(file_name + '_k.pickle', 'rb') as handle:
        ks = np.array(pickle.load(handle))

    return ks


def get_USGS_data(endmember, CRISM_match=False):
    """
    Get USGS spectral reflectance data for this endmember, as Pandas DataFrame
    If CRISM_match = True, then keep only reflectance values for wavelengths that we determined to keep; those in RW_USGS.  
    """
    # Normalize filename
    endmember = endmember.lower()
    file_name = USGS_DATA + endmember + ".csv"

    for r in ["(", ")", " "]:
        file_name = file_name.replace(r, "")

    # Open data in Pandas DataFrame
    data = pd.read_csv(file_name)

    # Replace NULL values (which are -1.23e34) with 0
    data.loc[data['reflectance'] < 0, 'reflectance'] = 0

    if CRISM_match:
        # Only keep rows with reduced wavelengths
        with open(MODULE_DIR + "/utils/FILE_CONSTANTS/RW_USGS.pickle", 'rb') as handle:
            RW_USGS = pickle.load(handle)
        data = data[data['wavelength'].isin(RW_USGS)]

    return data


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
