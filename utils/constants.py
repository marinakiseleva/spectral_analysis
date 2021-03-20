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
ENDMEMBERS_K = DATA_DIR + "lab_spectra/derived/"


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


""" 
Segmentation model parameters

MAX_SAD = 0.01; initial sad to allow merging of clusters
SEG_BURN_IN, SEG_EARLY_STOP - for segmentation
"""
DISTANCE_METRIC = 'SAD'
MAX_SAD = 0.01
SEG_BURN_IN = 10000
SEG_EARLY_STOP = 400


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


INITIAL_D = 80

# Min and max grain sizes for these endmembers
# Used to generate synthetic data and in inference (D prior)
GRAIN_SIZE_MIN = 45
GRAIN_SIZE_MAX = 100
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


"""
N_WAVELENGTHS
normalized wavelengths
USGS, RELAB, and CRISM wavelengths all matched to one another. These are the USGS wavelengths that are kept. 
"""
N_WAVELENGTHS = [0.43613,
                 0.44263,
                 0.44914,
                 0.45564,
                 0.46215,
                 0.46865,
                 0.47516,
                 0.48167,
                 0.48817,
                 0.49468,
                 0.50119,
                 0.5077,
                 0.51421,
                 0.52072,
                 0.52723,
                 0.53374,
                 0.54025,
                 0.54676,
                 0.55327,
                 0.55978,
                 0.56629,
                 0.57281,
                 0.57932,
                 0.58583,
                 0.59235,
                 0.59886,
                 0.60538,
                 0.61189,
                 0.61841,
                 0.62492,
                 0.63144,
                 0.70968,
                 0.7162,
                 0.72272,
                 0.72925,
                 0.73577,
                 0.7423,
                 0.74882,
                 0.75535,
                 0.76187,
                 0.7684,
                 0.77492,
                 0.78145,
                 0.78798,
                 0.79451,
                 0.80104,
                 0.80756,
                 0.81409,
                 0.82062,
                 0.82715,
                 0.83368,
                 0.84022,
                 0.84675,
                 0.85328,
                 0.85981,
                 0.86634,
                 0.87288,
                 0.87941,
                 0.88595,
                 0.89248,
                 0.89902,
                 0.90555,
                 0.91209,
                 0.91862,
                 0.92516,
                 0.9317,
                 0.93824,
                 0.94477,
                 0.95131,
                 0.95785,
                 0.96439,
                 0.97093,
                 0.97747,
                 0.98401,
                 0.99055,
                 0.9971,
                 1.00364,
                 1.01018,
                 1.0472,
                 1.05375,
                 1.0603,
                 1.06685,
                 1.07341,
                 1.07996,
                 1.08651,
                 1.09307,
                 1.09962,
                 1.10617,
                 1.11273,
                 1.11928,
                 1.12584,
                 1.13239,
                 1.13895,
                 1.14551,
                 1.15206,
                 1.15862,
                 1.16518,
                 1.17173,
                 1.17829,
                 1.18485,
                 1.19141,
                 1.19797,
                 1.20453,
                 1.21109,
                 1.21765,
                 1.22421,
                 1.23077,
                 1.23733,
                 1.24389,
                 1.25045,
                 1.25701,
                 1.26357,
                 1.27014,
                 1.2767,
                 1.28326,
                 1.28983,
                 1.29639,
                 1.30295,
                 1.30952,
                 1.31608,
                 1.32265,
                 1.32921,
                 1.33578,
                 1.34234,
                 1.34891,
                 1.35548,
                 1.36205,
                 1.36861,
                 1.37518,
                 1.38175,
                 1.38832,
                 1.39489,
                 1.40145,
                 1.40802,
                 1.41459,
                 1.42116,
                 1.42773,
                 1.43431,
                 1.44088,
                 1.44745,
                 1.45402,
                 1.46059,
                 1.46716,
                 1.47374,
                 1.48031,
                 1.48688,
                 1.49346,
                 1.50003,
                 1.50661,
                 1.51318,
                 1.51976,
                 1.52633,
                 1.53291,
                 1.53948,
                 1.54606,
                 1.55264,
                 1.55921,
                 1.56579,
                 1.57237,
                 1.57895,
                 1.58552,
                 1.5921,
                 1.59868,
                 1.60526,
                 1.61184,
                 1.61842,
                 1.625,
                 1.63158,
                 1.63816,
                 1.64474,
                 1.65133,
                 1.65791,
                 1.66449,
                 1.67107,
                 1.67766,
                 1.68424,
                 1.69082,
                 1.69741,
                 1.70399,
                 1.71058,
                 1.71716,
                 1.72375,
                 1.73033,
                 1.73692,
                 1.74351,
                 1.75009,
                 1.75668,
                 1.76327,
                 1.76985,
                 1.77644,
                 1.78303,
                 1.78962,
                 1.79621,
                 1.8028,
                 1.80939,
                 1.81598,
                 1.82257,
                 1.82916,
                 1.83575,
                 1.84234,
                 1.84893,
                 1.85552,
                 1.86212,
                 1.86871,
                 1.8753,
                 1.8819,
                 1.88849,
                 1.89508,
                 1.90168,
                 1.90827,
                 1.91487,
                 1.92146,
                 1.92806,
                 1.93465,
                 1.94125,
                 1.94785,
                 1.95444,
                 1.96104,
                 1.96764,
                 1.97424,
                 1.98084,
                 1.98743,
                 1.99403,
                 2.00063,
                 2.00723,
                 2.01383,
                 2.02043,
                 2.02703,
                 2.03363,
                 2.04024,
                 2.04684,
                 2.05344,
                 2.06004,
                 2.06664,
                 2.07325,
                 2.07985,
                 2.08645,
                 2.09306,
                 2.09966,
                 2.10627,
                 2.11287,
                 2.11948,
                 2.12608,
                 2.13269,
                 2.1393,
                 2.1459,
                 2.15251,
                 2.15912,
                 2.16572,
                 2.17233,
                 2.17894,
                 2.18555,
                 2.19216,
                 2.19877,
                 2.20538,
                 2.21199,
                 2.2186,
                 2.22521,
                 2.23182,
                 2.23843,
                 2.24504,
                 2.25165,
                 2.25827,
                 2.26488,
                 2.27149,
                 2.2781,
                 2.28472,
                 2.29133,
                 2.29795,
                 2.30456,
                 2.31118,
                 2.31779,
                 2.32441,
                 2.33102,
                 2.33764,
                 2.34426,
                 2.35087,
                 2.35749,
                 2.36411,
                 2.37072,
                 2.37734,
                 2.38396,
                 2.39058,
                 2.3972,
                 2.40382,
                 2.41044,
                 2.41706,
                 2.42368,
                 2.4303,
                 2.43692,
                 2.44354,
                 2.45017,
                 2.45679,
                 2.46341,
                 2.47003,
                 2.47666,
                 2.48328,
                 2.4899,
                 2.49653,
                 2.50312,
                 2.50972,
                 2.51632,
                 2.52292,
                 2.52951,
                 2.53611,
                 2.54271,
                 2.54931,
                 2.55591,
                 2.56251,
                 2.56911,
                 2.57571,
                 2.58231,
                 2.58891,
                 2.59551,
                 2.60212,
                 2.60872,
                 2.61532,
                 2.62192,
                 2.62853,
                 2.63513,
                 2.64174,
                 2.64834,
                 2.65495,
                 2.80035,
                 2.80697,
                 2.81358,
                 2.8202,
                 2.82681,
                 2.83343,
                 2.84004,
                 2.84666,
                 2.85328,
                 2.85989,
                 2.86651,
                 2.87313,
                 2.87975,
                 2.88636,
                 2.89298,
                 2.8996,
                 2.90622,
                 2.91284,
                 2.91946,
                 2.92608,
                 2.9327,
                 2.93932,
                 2.94595,
                 2.95257,
                 2.95919,
                 2.96581,
                 2.97244,
                 2.97906,
                 2.98568,
                 2.99231,
                 2.99893,
                 3.00556,
                 3.01218,
                 3.01881,
                 3.02544,
                 3.03206,
                 3.03869,
                 3.04532,
                 3.05195,
                 3.05857,
                 3.0652,
                 3.07183,
                 3.07846,
                 3.08509,
                 3.09172,
                 3.09835,
                 3.10498,
                 3.11161,
                 3.11825,
                 3.12488,
                 3.13151,
                 3.13814,
                 3.14478,
                 3.15141,
                 3.15804,
                 3.16468,
                 3.17131,
                 3.17795,
                 3.18458,
                 3.19122,
                 3.19785,
                 3.20449,
                 3.21113,
                 3.21776,
                 3.2244,
                 3.23104,
                 3.23768,
                 3.24432,
                 3.25096,
                 3.2576,
                 3.26424,
                 3.27088,
                 3.27752,
                 3.28416,
                 3.2908,
                 3.29744,
                 3.30408,
                 3.31073,
                 3.31737,
                 3.32401,
                 3.33066,
                 3.3373,
                 3.34395,
                 3.35059,
                 3.35724,
                 3.36388,
                 3.37053,
                 3.37717,
                 3.38382,
                 3.39047,
                 3.39712,
                 3.40376,
                 3.41041,
                 3.41706,
                 3.42371,
                 3.43036,
                 3.43701,
                 3.44366,
                 3.45031,
                 3.45696,
                 3.46361,
                 3.47026,
                 3.47692,
                 3.48357,
                 3.49022,
                 3.49687,
                 3.50353,
                 3.51018,
                 3.51684,
                 3.52349,
                 3.53015,
                 3.5368,
                 3.54346,
                 3.55011,
                 3.55677,
                 3.56343,
                 3.57008,
                 3.57674,
                 3.5834,
                 3.59006,
                 3.59672,
                 3.60338,
                 3.61004,
                 3.6167,
                 3.62336,
                 3.63002,
                 3.63668,
                 3.64334,
                 3.65,
                 3.65667,
                 3.66333,
                 3.66999,
                 3.67665,
                 3.68332,
                 3.68998,
                 3.69665,
                 3.70331,
                 3.70998,
                 3.71664,
                 3.72331,
                 3.72998,
                 3.73664,
                 3.74331,
                 3.74998,
                 3.75665,
                 3.76331,
                 3.76998,
                 3.77665,
                 3.78332,
                 3.78999,
                 3.79666,
                 3.80333,
                 3.81,
                 3.81667,
                 3.82335,
                 3.83002,
                 3.83669,
                 3.84336,
                 3.85004,
                 3.85671,
                 3.86339,
                 3.87006,
                 3.87673,
                 3.88341,
                 3.89008,
                 3.89676]

"""
c_wavelengths:
Wavelengths all endmembers have in common. Endmembers have different wavelengths available so we use the maximal set of wavelengths common to all endmembers (Note: olivine and anorthite have only these, and enstatite has more.)
"""
c_wavelengths = [0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1., 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28, 1.29, 1.3, 1.31, 1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.38, 1.39, 1.4, 1.41, 1.42, 1.43,
                 1.44, 1.45, 1.46, 1.47, 1.48, 1.49, 1.5, 1.51, 1.52, 1.53, 1.54, 1.55, 1.56, 1.57, 1.58, 1.59, 1.6, 1.61, 1.62, 1.63, 1.64, 1.65, 1.66, 1.67, 1.68, 1.69, 1.7, 1.71, 1.72, 1.73, 1.74, 1.75, 1.76, 1.77, 1.78, 1.79, 1.8, 1.81, 1.82, 1.83, 1.84, 1.85, 1.86, 1.87, 1.88, 1.89, 1.9, 1.91, 1.92, 1.93, 1.94, 1.95, 1.96, 1.97, 1.98, 1.99, 2., 2.01, 2.02, 2.03, 2.04, 2.05, 2.06, 2.07, 2.08, 2.09, 2.1, 2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17, 2.18, 2.19, 2.2, 2.21, 2.22, 2.23, 2.24, 2.25, 2.26, 2.27, 2.28, 2.29, 2.3, 2.31, 2.32, 2.33, 2.34, 2.35, 2.36, 2.37, 2.38, 2.39, 2.4, 2.41, 2.42, 2.43, 2.44, 2.45, 2.46, 2.47, 2.48, 2.49, 2.5]

enstatite_k = [1.51496223e-04, 1.36590770e-04, 1.24747244e-04, 1.17885588e-04, 1.04959127e-04, 9.83286161e-05, 9.45474361e-05, 9.14028231e-05, 8.63752593e-05, 7.66511800e-05, 7.31103124e-05, 6.86765339e-05, 6.34582403e-05, 6.14579240e-05, 6.26091729e-05, 6.53082655e-05, 6.20494487e-05, 6.11459735e-05, 6.13109260e-05, 6.14027585e-05, 6.13476425e-05, 6.15868363e-05, 6.21237885e-05, 6.23660085e-05, 6.24781206e-05, 6.29662734e-05, 6.29097540e-05, 6.25904343e-05, 6.33823036e-05, 6.34392476e-05, 6.52496439e-05, 6.79811213e-05, 7.19380348e-05, 7.93595134e-05, 8.99372366e-05, 1.07021318e-04, 1.28961805e-04, 1.62099658e-04, 2.06206876e-04, 2.64681890e-04, 3.35795801e-04, 4.17057756e-04, 5.19382312e-04, 6.30373570e-04, 7.50116557e-04, 8.68879178e-04, 9.89122207e-04, 1.09541932e-03, 1.19047631e-03, 1.27991483e-03, 1.34431574e-03, 1.38061123e-03, 1.38475013e-03, 1.35076969e-03, 1.30897445e-03, 1.22959163e-03, 1.11527117e-03, 9.91790546e-04, 8.54947249e-04, 7.21271183e-04, 6.01073788e-04, 4.92432744e-04, 3.99582092e-04, 3.19230987e-04, 2.58341467e-04, 2.12028023e-04, 1.77277038e-04, 1.49917302e-04, 1.30359039e-04, 1.15510277e-04, 1.04582783e-04, 9.62320712e-05, 9.09116597e-05, 8.66601384e-05, 8.36776020e-05, 8.08703044e-05, 7.79702487e-05, 7.62849427e-05, 7.50392963e-05, 7.46807605e-05, 7.32198182e-05, 7.18519509e-05, 7.18519509e-05, 7.16586385e-05, 7.06152480e-05, 7.05307470e-05, 7.04041851e-05, 7.03831135e-05, 7.19811154e-05, 7.34833018e-05, 7.41239748e-05, 7.52642548e-05, 7.78769464e-05, 8.13072180e-05, 8.43818927e-05, 8.99103189e-05, 9.80347202e-05, 1.02721208e-04, 1.12070290e-04, 1.58121888e-04, 1.41756021e-04, 1.38858166e-04, 1.47690205e-04, 1.57366385e-04, 1.69950444e-04,
               1.81736682e-04, 1.96152084e-04, 2.10888666e-04, 2.29944812e-04, 2.48704680e-04, 2.68512365e-04, 2.92250093e-04, 3.16945806e-04, 3.41983647e-04, 3.72328010e-04, 3.98745695e-04, 4.31406167e-04, 4.63817001e-04, 4.95983187e-04, 5.32926439e-04, 5.70739019e-04, 6.14536787e-04, 6.54211475e-04, 7.00839315e-04, 7.38752026e-04, 7.84330214e-04, 8.31723935e-04, 8.76979476e-04, 9.19178027e-04, 9.70643768e-04, 1.01279152e-03, 1.06247878e-03, 1.10298708e-03, 1.14950431e-03, 1.19583367e-03, 1.23217078e-03, 1.27303706e-03, 1.31015046e-03, 1.34552351e-03, 1.37072827e-03, 1.41407194e-03, 1.43711517e-03, 1.46228374e-03, 1.48300191e-03, 1.49682749e-03, 1.50131480e-03, 1.49951826e-03, 1.51758076e-03, 1.49324731e-03, 1.49280039e-03, 1.50221387e-03, 1.49414155e-03, 1.47635803e-03, 1.44185500e-03, 1.40984540e-03, 1.36990789e-03, 1.33509219e-03, 1.30194053e-03, 1.26316771e-03, 1.22152081e-03, 1.17244017e-03, 1.12701714e-03, 1.07753237e-03, 1.03392764e-03, 9.85575586e-04, 9.38079666e-04, 8.89671119e-04, 8.43255650e-04, 8.00219319e-04, 7.54169170e-04, 7.10769064e-04, 6.71070685e-04, 6.33020849e-04, 5.94631268e-04, 5.64959766e-04, 5.32766937e-04, 5.00008042e-04, 4.74489670e-04, 4.45580821e-04, 4.21576359e-04, 4.00779989e-04, 3.82953313e-04, 3.65591113e-04, 3.50167183e-04, 3.30807385e-04, 3.16282382e-04, 3.04666595e-04, 3.01040403e-04, 3.08244183e-04, 3.46413969e-04, 3.36198109e-04, 3.90126914e-04, 3.56726624e-04, 2.94621674e-04, 2.60593848e-04, 2.47294201e-04, 2.39499049e-04, 2.41949003e-04, 2.55802084e-04, 2.65873007e-04, 2.46407497e-04, 2.28640727e-04, 2.25514012e-04, 2.28709178e-04, 2.29944812e-04, 2.34673313e-04, 2.40865067e-04, 2.43839394e-04, 2.37428996e-04, 2.40865067e-04, 2.45523972e-04]

anorthite_k = [3.15863541e-05, 2.97490755e-05, 2.97490755e-05, 2.71915794e-05, 2.63889081e-05, 2.56099310e-05, 2.48539486e-05, 2.34082728e-05, 2.27172813e-05, 2.20466874e-05, 2.20466874e-05, 2.13958887e-05, 2.13958887e-05, 2.13958887e-05, 2.07643011e-05, 2.07643011e-05, 2.07643011e-05, 2.07643011e-05, 2.07643011e-05, 2.13958887e-05, 2.13958887e-05, 2.20466874e-05, 2.20466874e-05, 2.27172813e-05, 2.27172813e-05, 2.34082728e-05, 2.34082728e-05, 2.41202821e-05, 2.41202821e-05, 2.41202821e-05, 2.48539486e-05, 2.48539486e-05, 2.48539486e-05, 2.48539486e-05, 2.56099310e-05, 2.56099310e-05, 2.56099310e-05, 2.48539486e-05, 2.56099310e-05, 2.56099310e-05, 2.56099310e-05, 2.63889081e-05, 2.71915794e-05, 2.71915794e-05, 2.80186656e-05, 2.88709092e-05, 2.88709092e-05, 2.97490755e-05, 3.06539530e-05, 3.06539530e-05, 3.06539530e-05, 3.15863541e-05, 3.15863541e-05, 3.25471161e-05, 3.35371015e-05, 3.35371015e-05, 3.35371015e-05, 3.45571994e-05, 3.56083255e-05, 3.56083255e-05, 3.78074666e-05, 3.89574562e-05, 4.01424249e-05, 4.13634368e-05, 4.26215883e-05, 4.39180089e-05, 4.66303493e-05, 4.80487044e-05, 5.10161531e-05, 5.25679112e-05, 5.41668691e-05, 5.58144625e-05, 5.75121707e-05, 5.92615181e-05, 6.10640754e-05, 6.29214611e-05, 6.48353429e-05, 6.68074392e-05, 6.88395207e-05, 6.88395207e-05, 7.09334120e-05, 7.30909933e-05, 7.30909933e-05, 7.53142017e-05, 7.53142017e-05, 7.53142017e-05, 7.53142017e-05, 7.53142017e-05, 7.76050334e-05, 7.76050334e-05, 7.99655453e-05, 7.76050334e-05, 7.76050334e-05, 7.76050334e-05, 7.76050334e-05, 7.99655453e-05, 7.99655453e-05, 7.99655453e-05, 7.76050334e-05, 8.23978568e-05, 8.49041520e-05, 8.49041520e-05, 8.23978568e-05, 8.23978568e-05, 7.99655453e-05,
               7.99655453e-05, 7.76050334e-05, 7.53142017e-05, 7.53142017e-05, 7.09334120e-05, 7.09334120e-05, 7.09334120e-05, 6.88395207e-05, 6.68074392e-05, 6.68074392e-05, 6.48353429e-05, 6.29214611e-05, 6.48353429e-05, 6.29214611e-05, 6.29214611e-05, 6.29214611e-05, 6.29214611e-05, 6.29214611e-05, 6.10640754e-05, 6.10640754e-05, 6.10640754e-05, 6.10640754e-05, 6.10640754e-05, 6.10640754e-05, 6.10640754e-05, 5.92615181e-05, 5.92615181e-05, 5.92615181e-05, 5.92615181e-05, 5.92615181e-05, 5.92615181e-05, 6.10640754e-05, 6.10640754e-05, 6.10640754e-05, 5.92615181e-05, 5.92615181e-05, 5.92615181e-05, 6.10640754e-05, 6.10640754e-05, 6.29214611e-05, 6.29214611e-05, 6.48353429e-05, 6.68074392e-05, 7.09334120e-05, 7.99655453e-05, 9.86265846e-05, 1.18051653e-04, 1.33083472e-04, 1.50029332e-04, 1.69132952e-04, 1.64140297e-04, 1.50029332e-04, 1.45600600e-04, 1.37131472e-04, 1.29154967e-04, 1.25342427e-04, 1.18051653e-04, 1.11184960e-04, 1.07902879e-04, 1.01626509e-04, 9.86265846e-05, 9.57152154e-05, 9.01477631e-05, 8.74866812e-05, 9.01477631e-05, 8.74866812e-05, 8.74866812e-05, 8.23978568e-05, 7.76050334e-05, 7.99655453e-05, 8.23978568e-05, 8.49041520e-05, 7.99655453e-05, 8.49041520e-05, 8.74866812e-05, 9.01477631e-05, 9.01477631e-05, 8.74866812e-05, 8.74866812e-05, 8.74866812e-05, 9.01477631e-05, 8.74866812e-05, 8.49041520e-05, 8.74866812e-05, 9.01477631e-05, 9.01477631e-05, 9.28897872e-05, 1.01626509e-04, 1.01626509e-04, 1.04717682e-04, 1.07902879e-04, 1.07902879e-04, 1.14566873e-04, 1.18051653e-04, 1.21642429e-04, 1.29154967e-04, 1.41302599e-04, 1.50029332e-04, 1.69132952e-04, 1.69132952e-04, 1.69132952e-04, 1.79578465e-04, 1.79578465e-04, 1.74277468e-04, 1.79578465e-04, 1.90669084e-04]


olivine_k = [9.57152154e-05, 8.74866812e-05, 8.23978568e-05, 7.99655453e-05, 7.76050334e-05, 7.53142017e-05, 7.30909933e-05, 6.88395207e-05, 6.48353429e-05, 6.29214611e-05, 5.75121707e-05, 5.10161531e-05, 4.66303493e-05, 4.39180089e-05, 4.13634368e-05, 4.13634368e-05, 4.13634368e-05, 4.26215883e-05, 4.39180089e-05, 4.66303493e-05, 4.80487044e-05, 5.10161531e-05, 5.41668691e-05, 5.58144625e-05, 5.75121707e-05, 5.75121707e-05, 5.75121707e-05, 5.75121707e-05, 5.75121707e-05, 6.10640754e-05, 6.48353429e-05, 6.88395207e-05, 7.30909933e-05, 7.99655453e-05, 9.01477631e-05, 1.01626509e-04, 1.14566873e-04, 1.33083472e-04, 1.54592774e-04, 1.79578465e-04, 2.02444651e-04, 2.35164288e-04, 2.73172160e-04, 2.98865287e-04, 3.36920571e-04, 3.68609536e-04, 4.03278998e-04, 4.41209286e-04, 4.68458012e-04, 4.97389596e-04, 5.28107971e-04, 5.60723488e-04, 5.77779012e-04, 5.95353313e-04, 6.32121848e-04, 6.51349095e-04, 6.71161177e-04, 6.91575883e-04, 7.12611543e-04, 7.12611543e-04, 7.34287045e-04, 7.56621850e-04, 7.79636013e-04, 7.79636013e-04, 7.79636013e-04, 7.79636013e-04, 7.79636013e-04, 7.56621850e-04, 7.34287045e-04, 7.12611543e-04, 6.91575883e-04, 6.51349095e-04, 6.32121848e-04, 6.13462172e-04, 5.95353313e-04, 5.60723488e-04, 5.60723488e-04, 5.44171429e-04, 5.28107971e-04, 5.28107971e-04, 5.12518693e-04, 5.12518693e-04, 4.97389596e-04, 4.97389596e-04, 4.82707097e-04, 4.68458012e-04, 4.68458012e-04, 4.54629547e-04, 4.41209286e-04, 4.28185180e-04, 4.15545533e-04, 4.03278998e-04, 3.91374560e-04, 3.68609536e-04, 3.57728510e-04, 3.36920571e-04, 3.26974974e-04, 3.17322963e-04, 2.98865287e-04, 2.90043049e-04, 2.65108360e-04, 2.57282597e-04, 2.42317279e-04, 2.28222447e-04, 2.21485523e-04,
             2.08602409e-04, 1.96468665e-04, 1.90669084e-04, 1.85040702e-04, 1.74277468e-04, 1.74277468e-04, 1.69132952e-04, 1.69132952e-04, 1.64140297e-04, 1.64140297e-04, 1.64140297e-04, 1.64140297e-04, 1.69132952e-04, 1.69132952e-04, 1.74277468e-04, 1.74277468e-04, 1.79578465e-04, 1.85040702e-04, 1.85040702e-04, 1.90669084e-04, 1.96468665e-04, 2.02444651e-04, 2.08602409e-04, 2.14947467e-04, 2.21485523e-04, 2.21485523e-04, 2.28222447e-04, 2.35164288e-04, 2.42317279e-04, 2.49687843e-04, 2.49687843e-04, 2.57282597e-04, 2.65108360e-04, 2.65108360e-04, 2.73172160e-04, 2.73172160e-04, 2.81481236e-04, 2.81481236e-04, 2.81481236e-04, 2.90043049e-04, 2.81481236e-04, 2.81481236e-04, 2.81481236e-04, 2.81481236e-04, 2.81481236e-04, 2.81481236e-04, 2.81481236e-04, 2.73172160e-04, 2.73172160e-04, 2.57282597e-04, 2.57282597e-04, 2.49687843e-04, 2.49687843e-04, 2.42317279e-04, 2.35164288e-04, 2.28222447e-04, 2.21485523e-04, 2.14947467e-04, 2.08602409e-04, 2.02444651e-04, 1.90669084e-04, 1.85040702e-04, 1.79578465e-04, 1.74277468e-04, 1.64140297e-04, 1.64140297e-04, 1.59295021e-04, 1.50029332e-04, 1.45600600e-04, 1.41302599e-04, 1.37131472e-04, 1.37131472e-04, 1.25342427e-04, 1.29154967e-04, 1.25342427e-04, 1.25342427e-04, 1.18051653e-04, 1.18051653e-04, 1.18051653e-04, 1.14566873e-04, 1.18051653e-04, 1.11184960e-04, 1.11184960e-04, 1.14566873e-04, 1.18051653e-04, 1.14566873e-04, 1.18051653e-04, 1.21642429e-04, 1.14566873e-04, 1.07902879e-04, 1.07902879e-04, 1.04717682e-04, 1.11184960e-04, 1.11184960e-04, 1.07902879e-04, 1.04717682e-04, 1.04717682e-04, 1.01626509e-04, 1.07902879e-04, 1.04717682e-04, 1.01626509e-04, 1.04717682e-04, 1.04717682e-04, 1.04717682e-04, 1.07902879e-04, 1.07902879e-04]
sids_k = {pure_olivine_sid: olivine_k,
          pure_enstatite_sid: enstatite_k,
          pure_anorthite_sid: anorthite_k}
