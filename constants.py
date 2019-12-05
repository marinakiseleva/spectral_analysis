import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/.."
CATALOGUE_PATH = ROOT_DIR + "/RelabDB2018Dec31/catalogues/"
DATA_PATH = ROOT_DIR + "/RelabDB2018Dec31/data/"

# Spectrum IDs
pure_olivine_sid = "C1PO17"  # Does not exist in ModalMineralogy
pure_enstatite_sid = "C2PE12"
pure_anorthite_sid = "C1PA12"


pure = [pure_olivine_sid, pure_enstatite_sid, pure_anorthite_sid]

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
all_sids = mixtures + pure


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

enstatite_k = [5.77779012e-04, 5.44171429e-04, 5.28107971e-04, 4.97389596e-04, 4.68458012e-04, 4.28185180e-04, 4.03278998e-04, 3.79821531e-04, 3.57728510e-04, 3.26974974e-04, 3.07955871e-04, 2.90043049e-04, 2.65108360e-04, 2.42317279e-04, 2.21485523e-04, 2.08602409e-04, 1.90669084e-04, 1.79578465e-04, 1.69132952e-04, 1.59295021e-04, 1.50029332e-04, 1.41302599e-04, 1.37131472e-04, 1.29154967e-04, 1.25342427e-04, 1.25342427e-04, 1.18051653e-04, 1.11184960e-04, 1.04717682e-04, 1.01626509e-04, 9.86265846e-05, 9.57152154e-05, 9.57152154e-05, 9.28897872e-05, 9.28897872e-05, 8.74866812e-05, 8.74866812e-05, 7.99655453e-05, 7.76050334e-05, 7.53142017e-05, 7.30909933e-05, 8.23978568e-05, 6.88395207e-05, 6.48353429e-05, 6.29214611e-05, 6.29214611e-05, 6.10640754e-05, 6.10640754e-05, 6.29214611e-05, 6.48353429e-05, 6.48353429e-05, 6.48353429e-05, 6.29214611e-05, 6.10640754e-05, 6.10640754e-05, 6.10640754e-05, 6.10640754e-05, 6.10640754e-05, 6.10640754e-05, 6.10640754e-05, 6.10640754e-05, 6.10640754e-05, 6.10640754e-05, 6.10640754e-05, 6.29214611e-05, 6.29214611e-05, 6.29214611e-05, 6.29214611e-05, 6.29214611e-05, 6.29214611e-05, 6.29214611e-05, 6.29214611e-05, 6.29214611e-05, 6.29214611e-05, 6.29214611e-05, 6.29214611e-05, 6.29214611e-05, 6.29214611e-05, 6.29214611e-05, 6.48353429e-05, 6.48353429e-05, 6.68074392e-05, 6.88395207e-05, 6.88395207e-05, 7.30909933e-05, 7.53142017e-05, 7.99655453e-05, 8.49041520e-05, 9.01477631e-05, 9.86265846e-05, 1.07902879e-04, 1.18051653e-04, 1.29154967e-04, 1.45600600e-04, 1.64140297e-04, 1.85040702e-04, 2.08602409e-04, 2.35164288e-04, 2.65108360e-04, 2.98865287e-04, 3.36920571e-04, 3.79821531e-04, 4.15545533e-04, 4.68458012e-04, 5.12518693e-04,
               5.77779012e-04, 6.32121848e-04, 6.91575883e-04, 7.56621850e-04, 8.03350198e-04, 8.78909065e-04, 9.33189772e-04, 9.90822810e-04, 1.05201522e-03, 1.08401436e-03, 1.15096220e-03, 1.18597101e-03, 1.22204469e-03, 1.25921561e-03, 1.29751717e-03, 1.33698374e-03, 1.33698374e-03, 1.37765077e-03, 1.37765077e-03, 1.37765077e-03, 1.37765077e-03, 1.33698374e-03, 1.33698374e-03, 1.29751717e-03, 1.25921561e-03, 1.22204469e-03, 1.18597101e-03, 1.11698682e-03, 1.05201522e-03, 9.90822810e-04, 9.05642838e-04, 8.52964450e-04, 7.79636013e-04, 7.12611543e-04, 6.51349095e-04, 5.95353313e-04, 5.44171429e-04, 4.97389596e-04, 4.41209286e-04, 4.03278998e-04, 3.57728510e-04, 3.17322963e-04, 2.90043049e-04, 2.57282597e-04, 2.35164288e-04, 2.14947467e-04, 1.96468665e-04, 1.79578465e-04, 1.64140297e-04, 1.50029332e-04, 1.41302599e-04, 1.29154967e-04, 1.21642429e-04, 1.14566873e-04, 1.07902879e-04, 1.04717682e-04, 1.01626509e-04, 9.57152154e-05, 9.28897872e-05, 9.01477631e-05, 8.74866812e-05, 8.74866812e-05, 8.49041520e-05, 8.49041520e-05, 8.23978568e-05, 7.99655453e-05, 7.99655453e-05, 7.76050334e-05, 7.76050334e-05, 7.76050334e-05, 7.53142017e-05, 7.53142017e-05, 7.53142017e-05, 7.53142017e-05, 7.30909933e-05, 7.30909933e-05, 7.30909933e-05, 7.30909933e-05, 7.30909933e-05, 7.30909933e-05, 7.09334120e-05, 7.09334120e-05, 7.09334120e-05, 7.09334120e-05, 7.09334120e-05, 7.09334120e-05, 7.09334120e-05, 7.09334120e-05, 7.09334120e-05, 7.09334120e-05, 7.09334120e-05, 7.30909933e-05, 7.30909933e-05, 7.30909933e-05, 7.30909933e-05, 7.53142017e-05, 7.53142017e-05, 7.53142017e-05, 7.76050334e-05, 7.76050334e-05, 7.99655453e-05, 8.23978568e-05, 8.23978568e-05, 8.49041520e-05, 8.74866812e-05, 9.01477631e-05]

anorthite_k = [3.15863541e-05, 2.97490755e-05, 2.97490755e-05, 2.71915794e-05, 2.63889081e-05, 2.56099310e-05, 2.48539486e-05, 2.34082728e-05, 2.27172813e-05, 2.20466874e-05, 2.20466874e-05, 2.13958887e-05, 2.13958887e-05, 2.13958887e-05, 2.07643011e-05, 2.07643011e-05, 2.07643011e-05, 2.07643011e-05, 2.07643011e-05, 2.13958887e-05, 2.13958887e-05, 2.20466874e-05, 2.20466874e-05, 2.27172813e-05, 2.27172813e-05, 2.34082728e-05, 2.34082728e-05, 2.41202821e-05, 2.41202821e-05, 2.41202821e-05, 2.48539486e-05, 2.48539486e-05, 2.48539486e-05, 2.48539486e-05, 2.56099310e-05, 2.56099310e-05, 2.56099310e-05, 2.48539486e-05, 2.56099310e-05, 2.56099310e-05, 2.56099310e-05, 2.63889081e-05, 2.71915794e-05, 2.71915794e-05, 2.80186656e-05, 2.88709092e-05, 2.88709092e-05, 2.97490755e-05, 3.06539530e-05, 3.06539530e-05, 3.06539530e-05, 3.15863541e-05, 3.15863541e-05, 3.25471161e-05, 3.35371015e-05, 3.35371015e-05, 3.35371015e-05, 3.45571994e-05, 3.56083255e-05, 3.56083255e-05, 3.78074666e-05, 3.89574562e-05, 4.01424249e-05, 4.13634368e-05, 4.26215883e-05, 4.39180089e-05, 4.66303493e-05, 4.80487044e-05, 5.10161531e-05, 5.25679112e-05, 5.41668691e-05, 5.58144625e-05, 5.75121707e-05, 5.92615181e-05, 6.10640754e-05, 6.29214611e-05, 6.48353429e-05, 6.68074392e-05, 6.88395207e-05, 6.88395207e-05, 7.09334120e-05, 7.30909933e-05, 7.30909933e-05, 7.53142017e-05, 7.53142017e-05, 7.53142017e-05, 7.53142017e-05, 7.53142017e-05, 7.76050334e-05, 7.76050334e-05, 7.99655453e-05, 7.76050334e-05, 7.76050334e-05, 7.76050334e-05, 7.76050334e-05, 7.99655453e-05, 7.99655453e-05, 7.99655453e-05, 7.76050334e-05, 8.23978568e-05, 8.49041520e-05, 8.49041520e-05, 8.23978568e-05, 8.23978568e-05, 7.99655453e-05,
               7.99655453e-05, 7.76050334e-05, 7.53142017e-05, 7.53142017e-05, 7.09334120e-05, 7.09334120e-05, 7.09334120e-05, 6.88395207e-05, 6.68074392e-05, 6.68074392e-05, 6.48353429e-05, 6.29214611e-05, 6.48353429e-05, 6.29214611e-05, 6.29214611e-05, 6.29214611e-05, 6.29214611e-05, 6.29214611e-05, 6.10640754e-05, 6.10640754e-05, 6.10640754e-05, 6.10640754e-05, 6.10640754e-05, 6.10640754e-05, 6.10640754e-05, 5.92615181e-05, 5.92615181e-05, 5.92615181e-05, 5.92615181e-05, 5.92615181e-05, 5.92615181e-05, 6.10640754e-05, 6.10640754e-05, 6.10640754e-05, 5.92615181e-05, 5.92615181e-05, 5.92615181e-05, 6.10640754e-05, 6.10640754e-05, 6.29214611e-05, 6.29214611e-05, 6.48353429e-05, 6.68074392e-05, 7.09334120e-05, 7.99655453e-05, 9.86265846e-05, 1.18051653e-04, 1.33083472e-04, 1.50029332e-04, 1.69132952e-04, 1.64140297e-04, 1.50029332e-04, 1.45600600e-04, 1.37131472e-04, 1.29154967e-04, 1.25342427e-04, 1.18051653e-04, 1.11184960e-04, 1.07902879e-04, 1.01626509e-04, 9.86265846e-05, 9.57152154e-05, 9.01477631e-05, 8.74866812e-05, 9.01477631e-05, 8.74866812e-05, 8.74866812e-05, 8.23978568e-05, 7.76050334e-05, 7.99655453e-05, 8.23978568e-05, 8.49041520e-05, 7.99655453e-05, 8.49041520e-05, 8.74866812e-05, 9.01477631e-05, 9.01477631e-05, 8.74866812e-05, 8.74866812e-05, 8.74866812e-05, 9.01477631e-05, 8.74866812e-05, 8.49041520e-05, 8.74866812e-05, 9.01477631e-05, 9.01477631e-05, 9.28897872e-05, 1.01626509e-04, 1.01626509e-04, 1.04717682e-04, 1.07902879e-04, 1.07902879e-04, 1.14566873e-04, 1.18051653e-04, 1.21642429e-04, 1.29154967e-04, 1.41302599e-04, 1.50029332e-04, 1.69132952e-04, 1.69132952e-04, 1.69132952e-04, 1.79578465e-04, 1.79578465e-04, 1.74277468e-04, 1.79578465e-04, 1.90669084e-04]


olivine_k = [9.57152154e-05, 8.74866812e-05, 8.23978568e-05, 7.99655453e-05, 7.76050334e-05, 7.53142017e-05, 7.30909933e-05, 6.88395207e-05, 6.48353429e-05, 6.29214611e-05, 5.75121707e-05, 5.10161531e-05, 4.66303493e-05, 4.39180089e-05, 4.13634368e-05, 4.13634368e-05, 4.13634368e-05, 4.26215883e-05, 4.39180089e-05, 4.66303493e-05, 4.80487044e-05, 5.10161531e-05, 5.41668691e-05, 5.58144625e-05, 5.75121707e-05, 5.75121707e-05, 5.75121707e-05, 5.75121707e-05, 5.75121707e-05, 6.10640754e-05, 6.48353429e-05, 6.88395207e-05, 7.30909933e-05, 7.99655453e-05, 9.01477631e-05, 1.01626509e-04, 1.14566873e-04, 1.33083472e-04, 1.54592774e-04, 1.79578465e-04, 2.02444651e-04, 2.35164288e-04, 2.73172160e-04, 2.98865287e-04, 3.36920571e-04, 3.68609536e-04, 4.03278998e-04, 4.41209286e-04, 4.68458012e-04, 4.97389596e-04, 5.28107971e-04, 5.60723488e-04, 5.77779012e-04, 5.95353313e-04, 6.32121848e-04, 6.51349095e-04, 6.71161177e-04, 6.91575883e-04, 7.12611543e-04, 7.12611543e-04, 7.34287045e-04, 7.56621850e-04, 7.79636013e-04, 7.79636013e-04, 7.79636013e-04, 7.79636013e-04, 7.79636013e-04, 7.56621850e-04, 7.34287045e-04, 7.12611543e-04, 6.91575883e-04, 6.51349095e-04, 6.32121848e-04, 6.13462172e-04, 5.95353313e-04, 5.60723488e-04, 5.60723488e-04, 5.44171429e-04, 5.28107971e-04, 5.28107971e-04, 5.12518693e-04, 5.12518693e-04, 4.97389596e-04, 4.97389596e-04, 4.82707097e-04, 4.68458012e-04, 4.68458012e-04, 4.54629547e-04, 4.41209286e-04, 4.28185180e-04, 4.15545533e-04, 4.03278998e-04, 3.91374560e-04, 3.68609536e-04, 3.57728510e-04, 3.36920571e-04, 3.26974974e-04, 3.17322963e-04, 2.98865287e-04, 2.90043049e-04, 2.65108360e-04, 2.57282597e-04, 2.42317279e-04, 2.28222447e-04, 2.21485523e-04,
             2.08602409e-04, 1.96468665e-04, 1.90669084e-04, 1.85040702e-04, 1.74277468e-04, 1.74277468e-04, 1.69132952e-04, 1.69132952e-04, 1.64140297e-04, 1.64140297e-04, 1.64140297e-04, 1.64140297e-04, 1.69132952e-04, 1.69132952e-04, 1.74277468e-04, 1.74277468e-04, 1.79578465e-04, 1.85040702e-04, 1.85040702e-04, 1.90669084e-04, 1.96468665e-04, 2.02444651e-04, 2.08602409e-04, 2.14947467e-04, 2.21485523e-04, 2.21485523e-04, 2.28222447e-04, 2.35164288e-04, 2.42317279e-04, 2.49687843e-04, 2.49687843e-04, 2.57282597e-04, 2.65108360e-04, 2.65108360e-04, 2.73172160e-04, 2.73172160e-04, 2.81481236e-04, 2.81481236e-04, 2.81481236e-04, 2.90043049e-04, 2.81481236e-04, 2.81481236e-04, 2.81481236e-04, 2.81481236e-04, 2.81481236e-04, 2.81481236e-04, 2.81481236e-04, 2.73172160e-04, 2.73172160e-04, 2.57282597e-04, 2.57282597e-04, 2.49687843e-04, 2.49687843e-04, 2.42317279e-04, 2.35164288e-04, 2.28222447e-04, 2.21485523e-04, 2.14947467e-04, 2.08602409e-04, 2.02444651e-04, 1.90669084e-04, 1.85040702e-04, 1.79578465e-04, 1.74277468e-04, 1.64140297e-04, 1.64140297e-04, 1.59295021e-04, 1.50029332e-04, 1.45600600e-04, 1.41302599e-04, 1.37131472e-04, 1.37131472e-04, 1.25342427e-04, 1.29154967e-04, 1.25342427e-04, 1.25342427e-04, 1.18051653e-04, 1.18051653e-04, 1.18051653e-04, 1.14566873e-04, 1.18051653e-04, 1.11184960e-04, 1.11184960e-04, 1.14566873e-04, 1.18051653e-04, 1.14566873e-04, 1.18051653e-04, 1.21642429e-04, 1.14566873e-04, 1.07902879e-04, 1.07902879e-04, 1.04717682e-04, 1.11184960e-04, 1.11184960e-04, 1.07902879e-04, 1.04717682e-04, 1.04717682e-04, 1.01626509e-04, 1.07902879e-04, 1.04717682e-04, 1.01626509e-04, 1.04717682e-04, 1.04717682e-04, 1.04717682e-04, 1.07902879e-04, 1.07902879e-04]
sids_k = {pure_olivine_sid: olivine_k,
          pure_enstatite_sid: enstatite_k,
          pure_anorthite_sid: anorthite_k}
