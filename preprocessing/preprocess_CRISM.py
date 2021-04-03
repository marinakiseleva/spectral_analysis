from preprocessing.CRISM_preprocess_helpers import *
from utils.constants import *
import pandas as pd

CUR_IMG_NAME = "hrl0000baba"


PIX_DIR = "/Users/marina/mars_data/bagnolddunes/pixels/"
CRISM_DATA_DIR = "/Users/marina/mars_data/bagnolddunes/mrocr_2102/trdr/2008/2008_202/hrl0000baba/"

# Save img to ../data/PREPROCESSED_DATA/
record_layered_data(img_dir=CRISM_DATA_DIR,
                    pixel_dir=PIX_DIR,
                    img_save_name=CUR_IMG_NAME)

record_reduced_spectra()

from preprocessing.estimatek import estimate_all_USGS_k
estimate_all_USGS_k()
