"""
Accesses RELAB and USGS data 
"""
import pickle
import pandas as pd
import numpy as np

from utils.constants import *


"""
USGS data access
"""


def reduce_wavelengths(short, long_w):
    """
    Reduce long list, long_w, to short. Each is a list of wavelengths, and we keep the wavelength in the long_w that is closest to the next value in the short list. 
    """
    long_w = w
    short = r
    s_index = 0
    new_long = []
    for l_index, l in enumerate(long_w):
        if s_index >= len(short):
            break
        cur_s = short[s_index]
        if l > cur_s:
            new_long.append(l)
            s_index += 1
    return new_long


def get_USGS_wavelengths(endmember=None):
    """
    Get wavelengths for the endmember as numpy vector
    """
    return np.array(USGS_REDUCED_WAVELENGTHS)


def get_USGS_endmember_k(endmember):
    """
    Get k as numpy vector for endmember
    """
    usgs_data = MODULE_DIR + "/output/data/derived/"
    file_name = usgs_data + endmember

    with open(file_name + '_k.pickle', 'rb') as handle:
        ks = np.array(pickle.load(handle))

    return ks


def get_USGS_data(endmember):
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

    # Only keep rows with reduced wavelengths (to match basaltic glass, which
    # comes from RELAB)
    data = data[data['wavelength'].isin(USGS_REDUCED_WAVELENGTHS)]

    print("For endmember : " + str(endmember))
    print(data.shape)

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


def get_RELAB_wavelengths(spectrum_id, spectra_db, cut=True):
    r_data = get_reflectance_data(spectrum_id, spectra_db, cut)
    return r_data['Wavelength(micron)'].values


def get_reflectance_spectra(spectrum_id, spectra_db, cut=True):
    r_data = get_reflectance_data(spectrum_id, spectra_db, cut)
    return r_data['Reflectance'].values


def get_reflectance_data(spectrum_id, spectra_db, cut):
    """
    Returns spectral reflectance for the passed-in spectrum ID
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

    if cut:
        reflectance_df = reflectance_df.loc[
            reflectance_df['Wavelength(micron)'].isin(c_wavelengths)]
    return reflectance_df
