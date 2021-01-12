"""
Run BEFORE run_crism_inference.py

This needs to be run for each new image or image subsection, to set up wavelengths files and k files.

This script:

1. Normalizes CRISM & USGS spectra by calling record_reduced_spectra in utils.access_data

record_reduced_spectra(wavelengths_file)

2. Computes k for each endmember (per wavelength). This uses the reduced wavelengths files produced by the previous command.


"""


from utils.access_data import *
from utils.constants import *
from preprocessing.estimatek import estimate_all_USGS_k


if __name__ == "__main__":

    IMG_DIR = DATA_DIR + 'GALE_CRATER/cartOrder/cartorder/'
    image_file = IMG_DIR + 'layered_img_sec_100_150.pickle'
    wavelengths_file = IMG_DIR + 'layered_wavelengths.pickle'

    # Normalize spectra across USGS, and CRISM per each CRISM image
    # (since different CRISM images have different wavelengths)
    record_reduced_spectra(wavelengths_file)

    estimate_all_USGS_k()
