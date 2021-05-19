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
ENDMEMBERS_K = DATA_DIR + "PREPROCESSED_DATA/FILE_CONSTANTS/"


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
RED = "#ff1a1a"
PURPLE = "#8c1aff"


"""
Model parameters

INF_BURN_IN and INF_EARLY_STOP are for pixel-independent models (pixel-independent and segmentation model.)
"""
NUM_CPUS = 8

INF_BURN_IN = 100
INF_EARLY_STOP = 400

D_PRIOR_COVARIANCE = 5
M_PRIOR_SCALE = 10

""" 
Segmentation model parameters

MAX_SAD initial SAD to allow merging of adjacent single-pixel clusters
SEG_BURN_IN, SEG_EARLY_STOP - for segmentation
"""
DISTANCE_METRIC = 'SAD'
MAX_SAD = 0.01  # 0.01
MAX_MERGE_SAD = 0.01
SEG_BURN_IN = 10000
SEG_EARLY_STOP = 1000


""" 
MRF Params

BETA: locality weight

MRF_EARLY_STOP: after burn-in, if average energy over last MRF_PREV_STEPS runs is greater than MRF_EARLY_STOP, we stop. Remember, average energy change should be negative.
"""
BETA = 5
MRF_BURN_IN = 100  # 200
MRF_PREV_STEPS = 400  # 50
MRF_EARLY_STOP = 20  # 30


#########################################
# Endmember constants/variables
#########################################


INITIAL_D = 100

# Min and max grain sizes for these endmembers
# Used to generate synthetic data and in inference (D prior)
GRAIN_SIZE_MIN = 40
GRAIN_SIZE_MAX = 120
# GRAIN_SIZE_MIN = 20  # 50
# GRAIN_SIZE_MAX = 350  # 800

# USGS endmembers for CRISM testing (and used to generate USGS synthetic data)
# **Exception: basaltic glass is RELAB; removed.
USGS_PURE_ENDMEMBERS = ['olivine (Fo51)',
                        # 'olivine (Fo80)',
                        'augite',
                        'labradorite',
                        'pigeonite',
                        'magnetite']
# 'basaltic glass']
USGS_NUM_ENDMEMBERS = len(USGS_PURE_ENDMEMBERS)


# Optical constant n per endmember
# from Lapotre DOI:10.1002/2016JE005133
ENDMEMBERS_N = {'olivine (Fo51)': 1.67,
                'olivine (Fo80)': 1.67,
                'augite': 1.7,  # pyroxene
                'labradorite': 1.7,  # plagioclase
                'pigeonite': 1.56,  # pyroxene
                'magnetite': 2.42,
                "C1PO17": 1.66,  # Pure RELAB olivine
                "C2PE12": 1.66,  # Pure RELAB enstatite
                "C1PA12": 1.57,  # Pure RELAB anorthite
                "C1BE100": 2.78,  # RELAB Basaltic glass
                # "basaltic glass": 2.78  # RELAB Basaltic glass, C1BE100
                }


#########################################
# USGS Endmembers
#########################################
# USGS cosine of incidence angle (deg)
USGS_COS_INCIDENCE_ANGLE = 1
# USGS cosine of emission angle (deg)
USGS_COS_EMISSION_ANGLE = 0.86602540378


# Grain Sizes, from Lapotre DOI:10.1002/2016JE005133
USGS_OLIVINE_Fo51_GS = 25
USGS_OLIVINE_Fo80_GS = 300
USGS_AUGITE_GS = 35
USGS_PIGEONITE_GS = 162
USGS_LABRADORITE_GS = 162
USGS_MAGNETITE_GS = 162
# USGS_BASALTIC_GLASS_GS = 60

# Name to type dict
USGS_GRAIN_SIZES = {'olivine (Fo51)': USGS_OLIVINE_Fo51_GS,
                    'olivine (Fo80)': USGS_OLIVINE_Fo80_GS,
                    'augite': USGS_AUGITE_GS,
                    'labradorite': USGS_LABRADORITE_GS,
                    'pigeonite': USGS_PIGEONITE_GS,
                    'magnetite': USGS_MAGNETITE_GS}
# 'basaltic glass': USGS_BASALTIC_GLASS_GS}


# Densities from http://webmineral.com/ and Lapotre DOI:10.1002/2016JE005133
USGS_densities = {'olivine (Fo51)': 3.32,
                  'olivine (Fo80)': 3.32,
                  'augite': 3.4,
                  'labradorite': 2.69,
                  'pigeonite': 3.38,
                  'magnetite': 5.15}
# 'basaltic glass': 2.78}


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
