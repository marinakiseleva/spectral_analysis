import os


# For data
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../.."
CATALOGUE_PATH = ROOT_DIR + "/RelabDB2018Dec31/catalogues/"
DATA_PATH = ROOT_DIR + "/RelabDB2018Dec31/data/"

# For plotting
FIG_WIDTH = 6
FIG_HEIGHT = 4
DPI = 240

#########################################
# Below are constants for the 3 pure endmembers:
# olivine
# enstatite
# anorthite
# Other minerals, even mixtures of these 3, may have different constants.
#########################################

# For endmember-specifics
NUM_ENDMEMBERS = 3

# Min and max grain sizes for these endmembers
GRAIN_SIZE_MIN = 45
GRAIN_SIZE_MAX = 75


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

olivine_enstatite_mix_sid5 = "CBXO15"
olivine_enstatite_mix_sid6 = "CBXO16"
olivine_enstatite_mix_sid7 = "CBXO17"
olivine_enstatite_mix_sid8 = "CBXO18"
olivine_enstatite_mix_sid9 = "CBXO19"

olivine_anorthite_mix_sid0 = "CBXO20"
olivine_anorthite_mix_sid1 = "CBXO21"
olivine_anorthite_mix_sid2 = "CBXO22"
olivine_anorthite_mix_sid3 = "CBXO23"
olivine_anorthite_mix_sid4 = "CBXO24"

enstatite_anorthite_mix_sid1 = "CBXA01"
enstatite_anorthite_mix_sid2 = "CBXA02"
enstatite_anorthite_mix_sid3 = "CBXA03"
enstatite_anorthite_mix_sid4 = "CBXA04"
enstatite_anorthite_mix_sid5 = "CBXA05"

ternary_mix_sid0 = "CMXO30"
ternary_mix_sid1 = "CMXO31"
ternary_mix_sid2 = "CMXO32"
ternary_mix_sid3 = "CMXO33"
ternary_mix_sid4 = "CMXO34"
ternary_mix_sid5 = "CMXO35"
ternary_mix_sid6 = "CMXO36"

mixtures = [olivine_enstatite_mix_sid5,
            olivine_enstatite_mix_sid6,
            olivine_enstatite_mix_sid7,
            olivine_enstatite_mix_sid8,
            olivine_enstatite_mix_sid9,
            olivine_anorthite_mix_sid0,
            olivine_anorthite_mix_sid1,
            olivine_anorthite_mix_sid2,
            olivine_anorthite_mix_sid3,
            olivine_anorthite_mix_sid4,
            enstatite_anorthite_mix_sid1,
            enstatite_anorthite_mix_sid2,
            enstatite_anorthite_mix_sid3,
            enstatite_anorthite_mix_sid4,
            enstatite_anorthite_mix_sid5,
            ternary_mix_sid0,
            ternary_mix_sid1,
            ternary_mix_sid2,
            ternary_mix_sid3,
            ternary_mix_sid4,
            ternary_mix_sid5,
            ternary_mix_sid6]
all_sids = mixtures + pure_endmembers


sids_n = {pure_olivine_sid: 1.66,
          pure_enstatite_sid: 1.66,
          pure_anorthite_sid: 1.57}


sids_densities = {pure_olivine_sid: 3.32,
                  pure_enstatite_sid: 3.2,
                  pure_anorthite_sid: 2.73}


# cosine of source_angle
mu_0 = 0.8660254037844387

# cosine of detect angle
mu = 1


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
