import pandas as pd
from constants import DATA_PATH, CATALOGUE_PATH, c_wavelengths


def get_data():
    """
    Pull down and merge spectral data sources, return as pandas DataFrame
    """

    file_name = CATALOGUE_PATH + "Minerals.xls"
    minerals = pd.read_excel(file_name)

    file_name = CATALOGUE_PATH + "Spectra_Catalogue.xls"
    Spectra_Catalogue = pd.read_excel(file_name)
    Spectra_Catalogue['SampleID'] = Spectra_Catalogue['SampleID'].str.strip()

    file_name = CATALOGUE_PATH + "Sample_Catalogue.xls"
    Sample_Catalogue = pd.read_excel(file_name)
    Sample_Catalogue['SampleID'] = Sample_Catalogue['SampleID'].str.strip()

    # file_name = CATALOGUE_PATH + "Chem_Analyses.xls"
    # Chem_Analyses = pd.read_excel(file_name)

    # file_name = CATALOGUE_PATH + "Modal_Mineralogy.xls"
    # Modal_Mineralogy = pd.read_excel(file_name)
    # Modal_Mineralogy['SampleID'] = Modal_Mineralogy['Sample ID'].str.strip()

    # All endmember samples are in 'sample_spectra_mixtures' except for pure
    # sample_spectra_mixtures = pd.merge(left=sample_spectra,
    #                                    right=Modal_Mineralogy,
    #                                    on='SampleID')

    sample_spectra = pd.merge(left=Spectra_Catalogue,
                              right=Sample_Catalogue,
                              on='SampleID')

    return sample_spectra


def get_grain_sizes(spectrum_id, sample_spectra):
    """
    Get range of grain sizes 
    :param spectrum_id: SpectrumID in dataset to look up
    :param sample_spectra:  Merge of Spectra_Catalogue and Sample_Catalogue
    """
    s = sample_spectra[sample_spectra['SpectrumID'] == spectrum_id]
    min_grain_size = s['MinSize'].values[0]
    max_grain_size = s['MaxSize'].values[0]
    return min_grain_size, max_grain_size


def get_wavelengths(spectrum_id, sample_spectra, cut=True):
    r_data = get_reflectance_data(spectrum_id, sample_spectra, cut)
    return r_data['Wavelength(micron)'].values


def get_reflectance_spectra(spectrum_id, sample_spectra, cut=True):
    r_data = get_reflectance_data(spectrum_id, sample_spectra, cut)
    return r_data['Reflectance'].values


def get_reflectance_data(spectrum_id, sample_spectra, cut):
    """
    Returns spectral reflectance for the passed-in spectrum ID
    :param spectrum_id: SpectrumID in dataset to look up
    :param sample_spectra:  Merge of Spectra_Catalogue and Sample_Catalogue
    :param cut: Boolean on whether to keep only wavelenghts in c_wavelengths, or to use all.
    :return reflectance_df: Pandas DataFrame with 2 columns [Wavelength(micron), Reflectance]
    """
    pi = sample_spectra[sample_spectra['SpectrumID'] == spectrum_id]["PI"].values[0]
    sampleid = sample_spectra[sample_spectra['SpectrumID']
                              == spectrum_id]["SampleID"].values[0]

    pi = pi.lower()
    pre_sampleid = sampleid[0:2].lower()
    spectrum_id = spectrum_id.lower()
    file_name = DATA_PATH + pi + "/" + pre_sampleid + "/" + spectrum_id + ".txt"

    reflectance_df = pd.read_csv(file_name, sep="\t", header=0, skiprows=1)

    if cut:
        reflectance_df = reflectance_df.loc[
            reflectance_df['Wavelength(micron)'].isin(c_wavelengths)]
    return reflectance_df
