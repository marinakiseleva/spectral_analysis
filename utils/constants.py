import os
import pickle

"""
DIRECTORIES
"""
# ROOT_DIR is one level above module root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../.."
# Module directory, contains model, preprocessing, etc.
MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/.."

DATA_DIR = ROOT_DIR + "/../data/"
CATALOGUE_PATH = DATA_DIR + "lab_spectra/RELAB/RelabDB2018Dec31/catalogues/"
RELAB_DATA_PATH = DATA_DIR + "lab_spectra/RELAB/RelabDB2018Dec31/data/"
USGS_DATA = DATA_DIR + "lab_spectra/USGS/"
PREPROCESSED_DATA = DATA_DIR + "PREPROCESSED_DATA/"
R_DIR = PREPROCESSED_DATA + "REFLECTANCE/"
K_DIR = PREPROCESSED_DATA + "K/"

# ENDMEMBERS_K = DATA_DIR + "PREPROCESSED_DATA/FILE_CONSTANTS/"


"""
For plotting
"""
FIG_WIDTH = 6
FIG_HEIGHT = 4
DPI = 240
LIGHT_GREEN = "#b3e6b3"
DARK_GREEN = "#339966"
LIGHT_BLUE = "#668cff"
DARK_BLUE = "#002699"
PINK = "#ff99c2"
LIGHT_ORANGE = "#ffa366"
LIGHT_RED = "#ff8566"
DARK_RED = "#801a00"
LIGHT_PURPLE = "#e699ff"
DARK_PURPLE = "#4d0066"
LIGHT_GRAY = "#ebebe0"
DARK_ORANGE = "#cc5200"



"""
Model parameters

INF_BURN_IN and INF_EARLY_STOP are for pixel-independent models (pixel-independent and segmentation model.)
"""
NUM_CPUS = 8

INF_EARLY_STOP = 200


""" 
Segmentation model parameters
SEG_BURN_IN, SEG_EARLY_STOP - for segmentation
"""
SEG_BURN_IN = 10000
SEG_EARLY_STOP = 1000


""" 
MRF Params
MRF_EARLY_STOP: after burn-in, if average energy over last MRF_PREV_STEPS runs is greater than MRF_EARLY_STOP, we stop.
Average energy change should be negative. Return MAP.
"""
MRF_BURN_IN = 100 
MRF_PREV_STEPS = 200  
MRF_EARLY_STOP = 20  


#########################################
# Endmember constants/variables
#########################################


INITIAL_D = 200
# Min and max grain sizes for these endmembers
# Used to generate synthetic data and in inference (D prior)
GRAIN_SIZE_MIN = 60
GRAIN_SIZE_MAX = 400 


# USGS endmembers for CRISM testing (and used to generate USGS synthetic data)
# **Exception: basaltic glass is RELAB; removed.
USGS_PURE_ENDMEMBERS = ["augite",
                        "enstatite", 
                        "labradorite",  
                        "olivine (Fo51)"]
# USGS_PURE_ENDMEMBERS = ['diopside',
#                         "augite",
#                         "pigeonite",
#                         "hypersthene", 
#                         "enstatite",
#                         "andesine",
#                         "labradorite", 
#                         "olivine (Fo51)",
                        # "magnetite"]
USGS_NUM_ENDMEMBERS = len(USGS_PURE_ENDMEMBERS)

# Calculated using Dale-Gladstone relationship
ENDMEMBERS_N = {'diopside':1.72,
                'augite': 1.68,   
                'pigeonite': 1.71,
                'hypersthene' : 1.69,
                'enstatite': 1.66,
                'andesine':1.47,
                'labradorite': 1.53, 
                'olivine (Fo51)': 1.66,
                'magnetite': 2.40} 

USGS_GRAIN_SIZES = {"diopside": 295,
                    "augite": 400,
                    "pigeonite":162,
                    "hypersthene":200, 
                    "enstatite":36,
                    "andesine":290,
                    "labradorite":162, 
                    "olivine (Fo51)":60, 
                    "magnetite":162}
# Densities from http://webmineral.com/ 
USGS_DENSITIES = {"diopside": 3.4,
                    "augite": 3.4,
                    "pigeonite":3.38,
                    "hypersthene":3.55, 
                    "enstatite":3.2,
                    "andesine":2.67,
                    "labradorite":2.69, 
                    "olivine (Fo51)":3.32, 
                    "magnetite":5.15} 

# USGS incidence angle : 0  (deg)
# USGS emission angle: 30 (deg)
USGS_COS_INCIDENCE_ANGLE = 1
USGS_COS_EMISSION_ANGLE = 0.86602540378

#########################################
# RELAB endmembers

# olivine
# enstatite
# anorthite
# Other minerals, even mixtures of these 3, may have different constants.
#########################################
RELAB_WAVELENGTH_COUNT = 211
# USGS incidence angle (deg)
RELAB_INCIDENCE_ANGLE = 30
# USGS emission angle (deg)
RELAB_EMISSION_ANGLE = 0

# Spectrum IDs
pure_olivine_sid = "C1PO17"  # Does not exist in ModalMineralogy
pure_enstatite_sid = "C2PE12"
pure_anorthite_sid = "C1PA12"

sids_names = {pure_olivine_sid: 'olivine',
              pure_enstatite_sid: 'enstatite',
              pure_anorthite_sid: 'anorthite'}

# pure_endmembers used elsewhere to preserve order of names in vectors and dicts.
pure_endmembers = [pure_olivine_sid,
                   pure_enstatite_sid,
                   pure_anorthite_sid]

all_sids = pure_endmembers


sids_densities = {pure_olivine_sid: 3.32,
                  pure_enstatite_sid: 3.2,
                  pure_anorthite_sid: 2.73}
