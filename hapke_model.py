import numpy as np
import math
from constants import *


def get_w_mixed_hapke_estimate(m, D):
    """
    Get estimated w mix 
    :param m: Map from SID to abundance
    :param D: Map from SID to grain size
    """
    sigmas = {}
    for endmember in m.keys():
        m_cur = m[endmember]
        D_cur = D[endmember]
        n = sids_n[endmember]
        k = sids_k[endmember]

        rho = sids_densities[endmember]

        sigmas[endmember] = m_cur / (rho * D_cur)

    sigma_sum = sum(sigmas.values())
    # F is the mapping of fractional abundances
    F = {k: v / sigma_sum for k, v in sigmas.items()}

    w_mix = np.zeros(len(c_wavelengths))
    for endmember in m.keys():
        D_cur = D[endmember]
        n = sids_n[endmember]
        k = sids_k[endmember]
        w = get_w_hapke_estimate(n, k, D_cur, np.array(c_wavelengths))
        w_mix = w_mix + (F[endmember] * w)

    r = get_derived_reflectance(w_mix, mu, mu_0)
    return r


def get_w_hapke_estimate(n, k, D, wavelengths):
    """
    Get w, SSA, as estimated by the Hapke model
    :param k: Numpy array, imaginary index of refraction
    :param n: scalar, real index of refraction
    :param D: scalar, grain size
    :param wavelengths: Numpy array, Wavelengths (lambda values)
    """
    Se = get_S_e(k, n)
    Si = get_S_i(n)
    brackets_D = get_brackets_D(n, D)
    Theta = get_Theta(k, wavelengths, brackets_D)

    return Se + (1 - Se) * ((1 - Si) / (1 - Si * Theta)) * Theta


def get_brackets_D(n, D):
    """
    Get the mean free path
    :param n: scalar, real index of refraction
    :param D: scalar, grain size
    """
    return (2 / 3) * ((n**2) - (1 / n) * ((n**2) - 1)**(3 / 2)) * D


def get_S_e(k, n):
    """
    Get S_e, surface reflection coefficient S_e for externally incident light
    :param k: Numpy array 
    :param n: scalar, real index of refraction
    """
    return ((((n - 1)**2) + k**2) / (((n + 1)**2) + k**2)) + 0.05


def get_S_i(n):
    """
    Get S_i, reflection coefficient for internally scattered light
    :param n: scalar, real index of refraction
    """
    return 1.014 - (4 / (n * ((n + 1)**2)))


def get_Theta(k, wavelengths, brackets_D):
    """
    :param k: scalar, k value for this wavelength
    :param wavelengths: Numpy array, wavelength values (lambda)
    :param brackets_D: scalar, <D>
    :return: Numpy array
    """
    alpha = get_alpha(k, wavelengths)
    return np.exp(- np.sqrt((alpha**2) * brackets_D))


def get_alpha(k, wavelengths):
    """
    Gets internal absorption coefficient
    :param k: scalar, k value for this wavelength
    :param wavelengths: Numpy array, wavelength values (lambda)
    :return: Numpy array
    """
    return (4 * math.pi * k) / wavelengths


def get_derived_reflectance(w, mu, mu_0):
    """
    Get reflectance from SSA, w
    """
    d = 4 * (mu + mu_0)
    return (w / d) * get_H(mu, w) * get_H(mu_0, w)


def get_reflectance_hapke_estimate(mu, mu_0, n, k, D, wavelengths):
    """
    Gets reflectance r(mu, mu_0, w)
    :param mu: cosine of detect angle
    :param mu_0: cosine of source angle 
    """
    w = get_w_hapke_estimate(n, k, D, wavelengths)
    return get_derived_reflectance(w, mu, mu_0)


def get_H(x, w):
    """
    get output of Chandrasekhar integral function given x and SSA w
    :param x: scalar value 
    :param w: Numpy array of SSA 
    """
    # Solve for gamma
    gamma = (1 - w) ** (1 / 2)
    # Solve for r_0
    r_0 = (1 - gamma) / (1 + gamma)

    # Inner fraction
    f = (1 - (2 * x * r_0)) / 2

    # Inner [r_0 + fraction * log]
    nf = r_0 + (f * math.log((1 + x) / x))

    # Denominator
    d = 1 - ((w * x) * nf)
    return 1 / d
