import numpy as np
import math
from utils.constants import *
from utils.access_data import get_USGS_endmember_k, get_USGS_wavelengths


def get_USGS_r_mixed_hapke_estimate(m, D):
    """
    Calculate reflectance of m and D using Hapke model; using spectral endmembers from USGS library
    :param m: Map from SID to abundance
    :param D: Map from SID to grain size
    """
    wavelengths = get_USGS_wavelengths(True)
    sigmas = {}
    for endmember in m.keys():
        m_cur = m[endmember]
        D_cur = D[endmember]
        rho = USGS_densities[endmember]
        sigmas[endmember] = m_cur / (rho * D_cur)
    sigma_sum = sum(sigmas.values())

    # F is the mapping of fractional abundances
    F = {s: v / sigma_sum for s, v in sigmas.items()}

    w_mix = np.zeros(len(wavelengths))
    for endmember in m.keys():
        D_cur = D[endmember]
        n = ENDMEMBERS_N[endmember]
        k = get_USGS_endmember_k(endmember)
        w = get_w_hapke_estimate(n, k, D_cur, wavelengths)
        # Gets mixture of SSAs of endmembers as single SSA:
        w_mix = w_mix + (F[endmember] * w)

    r = get_derived_reflectance(w_mix, mu, mu_0)
    return r


def get_synthetic_r_mixed_hapke_estimate(m, D):
    """
    Calculate reflectance of m and D using Hapke model; using spectral endmembers from RELAB library
    :param m: Map from SID to abundance
    :param D: Map from SID to grain size
    """
    sigmas = {}
    for endmember in m.keys():
        m_cur = m[endmember]
        D_cur = D[endmember]
        rho = sids_densities[endmember]
        sigmas[endmember] = m_cur / (rho * D_cur)

    sigma_sum = sum(sigmas.values())
    # F is the mapping of fractional abundances
    F = {s: v / sigma_sum for s, v in sigmas.items()}

    w_mix = np.zeros(len(c_wavelengths))
    for endmember in m.keys():
        D_cur = D[endmember]
        n = ENDMEMBERS_N[endmember]
        k = np.array(sids_k[endmember])
        w = get_w_hapke_estimate(n, k, D_cur, np.array(c_wavelengths))

        w_mix = w_mix + (F[endmember] * w)

    r = get_derived_reflectance(w_mix, mu, mu_0)
    return r


def get_reflectance_hapke_estimate(mu, mu_0, n, k, D, wavelengths):
    """
    Gets reflectance of SSA estimated from Hapke model (first gets SSA, then gets reflectance)
    :param mu: cosine of detect angle
    :param mu_0: cosine of source angle 
    :param n: real index of refraction (sclar)
    :param k: imaginary index of refraction (scalar)
    :param D: grain size (scalar)
    :param wavelengths: lambdas/wavelengths of data (Numpy array)
    :return reflectance: as Numpy array
    """
    w = get_w_hapke_estimate(n, k, D, wavelengths)
    return get_derived_reflectance(w, mu, mu_0)


def get_derived_reflectance(w, mu, mu_0):
    """
    Get reflectance from SSA
    :param w: SSA (Numpy array)
    :param mu: cosine of detect angle
    :param mu_0: cosine of source angle
    """
    d = 4 * (mu + mu_0)
    return (w / d) * get_H(mu, w) * get_H(mu_0, w)


def get_w_hapke_estimate(n, k, D, wavelengths):
    """
    Get SSA for particular endmember as estimated by the Hapke model:
    w = S_e + (1-S_e) * (1-S_i)*Theta/(1- Theta*S_i )
    :param k: imaginary index of refraction (scalar)
    :param n: real index of refraction (sclar)
    :param D: grain size (scalar)
    :param wavelengths: lambdas/wavelengths of data (Numpy array)
    """
    Se = get_S_e(k, n)
    Si = get_S_i(n)
    brackets_D = get_brackets_D(n, D)
    Theta = get_Theta(k, wavelengths, brackets_D)

    return Se + (1 - Se) * ((1 - Si) / (1 - Si * Theta)) * Theta


def get_brackets_D(n, D):
    """
    Get the mean free path
    :param n: real index of refraction (sclar)
    :param D: grain size (scalar)
    """
    return (2 / 3) * ((n**2) - (1 / n) * ((n**2) - 1)**(3 / 2)) * D


def get_S_e(k, n):
    """
    Get S_e, surface reflection coefficient S_e for externally incident light
    :param k: imaginary index of refraction (scalar)
    :param n: real index of refraction (sclar)
    """
    return ((((n - 1)**2) + k**2) / (((n + 1)**2) + k**2)) + 0.05


def get_S_i(n):
    """
    Get S_i, reflection coefficient for internally scattered light
    :param n: real index of refraction (sclar)
    """
    return 1.014 - (4 / (n * ((n + 1)**2)))


def get_Theta(k, wavelengths, brackets_D):
    """
    Gets the reflection coefficient for internally scattered light:
              r_i  + exp(- sqrt(alpha(alpha+s) <D> ))
     Theta =  --------------------------------------
              1 + r_i exp(- sqrt(alpha(alpha+s) <D> ))
    Which simplifies when s = 0 to:

     Theta = exp(- sqrt(alpha^2 <D>))
    :param k: imaginary index of refraction (scalar)
    :param wavelengths: Numpy array, wavelength values (lambda)
    :param brackets_D: scalar, <D>
    :return: Theta as Numpy array
    """
    alpha = get_alpha(k, wavelengths)
    return np.exp(-1 * np.sqrt((alpha**2) * brackets_D))


def get_alpha(k, wavelengths):
    """
    Gets internal absorption coefficient:
    alpha = 4*pi*k/ lambda
    :param k: imaginary index of refraction (scalar)
    :param wavelengths: lambdas/wavelengths of data (Numpy array)
    :return: alpha as Numpy array
    """
    return (4 * math.pi * k) / wavelengths


def get_H(x, w):
    """
    Gets Chandrasekhar integral function given x and SSA w
    H(x) = 1 / ( 1 - wx(r_0 + (1-2r_0x)/2 ln((1+x)/x) ))
    :param x: scalar value 
    :param w: Numpy array of SSA 
    """
    # Solve for gamma
    gamma = (1 - w) ** (1 / 2)

    # Solve for r_0
    r_0 = get_r_0(gamma)

    # Inner fraction
    f = (1 - (2 * x * r_0)) / 2

    # Inner [r_0 + fraction * log]
    nf = r_0 + (f * math.log((1 + x) / x))

    # Denominator
    d = 1 - ((w * x) * nf)
    return 1 / d


def get_r_0(gamma):
    """
    Gets the bihemispherical reflectance for isotropic scatterers:
    r_0 = (1-gamma)/(1+gamma)
    :param gamma: scalar value
    """
    return (1 - gamma) / (1 + gamma)
