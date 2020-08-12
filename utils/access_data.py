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


def match_lists(target, source):
    """
    Makes target list of wavelengths as similar to source as possible. For each target wavlength, finds closest source value.
    :param target: List of wavelengths
    :param source: List of wavelengths, longer than target
    Reduce long list, long_w, to short. Each is a list of wavelengths, and we keep the wavelength in the long_w that is closest to the next value in the short list.
    """

    # Finds first value in target that is greater than source - if necessary
    t_index = 0  # Starting index for target - where it reaches min value of source
    t_start = target[0]
    if min(target) < min(source):
        while t_start <= source[0]:
            t_index += 1
            t_start = target[t_index]
    if t_index > 0:
        print("T start reset to: " + str(t_start))
    # Reset target so next part works.
    target = target[t_index:]

    # Assumes there are more values in source than target - iterates over
    # source, keeping each value that is >= to next target value
    source_index = 0
    target_index = 0
    new_target = []
    new_source = []

    while source_index <= len(source) - 1:
        source_val = source[source_index]
        if target_index >= len(target):
            break
        targ_val = target[target_index]
        increment_source = True
        if source_val >= targ_val:
            if target_index == len(target) - 1:
                print("not adding " + str(source_val))
            elif source_val <= target[target_index + 1]:
                new_target.append(targ_val)
                new_source.append(source_val)
            else:
                increment_source = False
            target_index += 1
        if increment_source:
            source_index += 1
    return new_target, new_source


def record_reduced_spectra(CRISM_w_file):
    """
    Saves the wavelengths that are found to be as equal as possible between RELAB (basaltic glass), USGS (olivine Fo80), and CRISM (random pixel from passed-in image) spectra.  The unique wavelengths are saved per data source because this is how their spectral values will be accessed.
    :param CRISM_w_file: Path and .txt file name of CRISM z-profile for a single pixel, saved from CAT ENVI
    """
    # Get data
    ss = get_data()
    BASALTIC_wavelengths = get_RELAB_wavelengths(
        spectrum_id='C1BE100', spectra_db=ss, CRISM_match=False)
    USGS_data = get_USGS_data("olivine (Fo80)", CRISM_match=False)
    USGS_wavelengths = USGS_data['wavelength'].values.tolist()
    CRISM_wavelengths = get_CRISM_wavelengths(CRISM_w_file)

    # Print current spec stats
    print("RELAB min:" + str(min(BASALTIC_wavelengths)) + " max: " +
          str(max(BASALTIC_wavelengths)) + " count: " + str(len(BASALTIC_wavelengths)))
    print("USGS min:" + str(min(USGS_wavelengths)) + " max: " +
          str(max(USGS_wavelengths)) + " count: " + str(len(USGS_wavelengths)))
    print("CRISM min:" + str(min(CRISM_wavelengths)) + " max: " +
          str(max(CRISM_wavelengths)) + " count: " + str(len(CRISM_wavelengths)))

    # 1. Reduce Olivine Fo51 (which is the same as all USGS endmembers) to basaltic glass
    # Reduce both source and target to match target as best as possible
    # (target has less values)
    basalt_rw, usgs_rw = match_lists(target=BASALTIC_wavelengths,
                                     source=USGS_wavelengths)
    # 2. Match USGS to CRISM
    CRISM_reduced, USGS_reduced = match_lists(target=CRISM_wavelengths, source=usgs_rw)
    print("CRISM_reduced min:" + str(min(CRISM_reduced)) + " max: " +
          str(max(CRISM_reduced)) + " count: " + str(len(CRISM_reduced)))

    # 3. Find same indexed wavelengths for basaltic glass
    # Get indices of original USGS that were kept
    keep_indices = [usgs_rw.index(i) for i in USGS_reduced]

    # Find these indices in basaltic glass spectra
    BASALT_reduced = np.take(a=basalt_rw, indices=keep_indices, axis=0)

    print("\nAll reduced spectra should have same length ")
    print("CRISM : " + str(len(CRISM_reduced)))
    print("USGS " + str(len(USGS_reduced)))
    print("BASALT " + str(len(BASALT_reduced)))

    path = MODULE_DIR + "/utils/FILE_CONSTANTS/"
    with open(path + 'RW_BASALT.pickle', 'wb') as f:
        pickle.dump(BASALT_reduced, f)
    with open(path + 'RW_USGS.pickle', 'wb') as f:
        pickle.dump(USGS_reduced, f)
    with open(path + 'RW_CRISM.pickle', 'wb') as f:
        pickle.dump(CRISM_reduced, f)


"""
CRISM data access
"""


def get_CRISM_data(image_file, wavelengths_file, CRISM_match):
    """
    Gets CRISM data this directory & name.
    :param image_file: Full dir and file name of Pickled image.
    :param wavelengths_file: Full dir and file name of wavelength values for image
    :param CRISM_match: filter spectra to same range as endmembers
    """

    with open(image_file, 'rb') as handle:
        loaded_img = pickle.load(handle)

    if CRISM_match:

        with open(MODULE_DIR + "/utils/FILE_CONSTANTS/RW_CRISM.pickle", 'rb') as handle:
            # CRISM reduced wavelengths to keep.
            RW_CRISM = pickle.load(handle)

        with open(wavelengths_file, 'rb') as handle:
            # CRISM wavelengths to keep.
            wavelengths = pickle.load(handle)
        keep_indices = []
        for index, w in enumerate(wavelengths):
            if w in RW_CRISM:
                keep_indices.append(index)

        newimg = np.zeros((loaded_img.shape[0], loaded_img.shape[1], len(RW_CRISM)))
        for xindex, row in enumerate(loaded_img):
            for yindex, val in enumerate(row):
                # Only keep values with reduced wavelengths
                newimg[xindex, yindex] = np.take(val, keep_indices)
        return newimg

    return loaded_img


def get_CRISM_wavelengths(file):
    """
    Get the wavelengths of CRISM data from saved file.
    :param file: Path and .pickle file name of CRISM wavelengths.
    """
    if file is None:
        return ValueError("Must pass in file.")

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
    Get USGS data as Pandas DataFrame 
    """
    file_name = USGS_DATA + endmember + '.txt'
    for r in ["(", ")", " "]:
        file_name = file_name.replace(r, "")
    data = pd.read_csv(file_name, sep='      ', skiprows=16, names=[
        'wavelength', 'reflectance', 'standard deviation'], engine='python')
    INVALID_VALUE = -1.23e34
    data.loc[data['reflectance'] == INVALID_VALUE, 'reflectance'] = 0

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
    Get RELAB data - Pull down and merge spectral data sources, return as pandas DataFrame
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
