"""
Runs variational inference on the model to estimate the posterior p(m,D|d)
"""
from scipy import stats
import numpy as np
import math


def sample_dirichlet(x):
    """
    Sample from dirichlet
    :param x: Vector that will be multiplied by constant and used as alpha parameter
    """
    c = 10
    return np.random.dirichlet(alpha=x * c)


def sample_multivariate(mean):
    """
    Sample from multivariate Gaussian
    :param mean: vector of mean of Gaussian
    """
    length = mean.shape[0]
    print("length of mean vector " + str(length))
    covariance = np.zeros((length, length))
    np.fill_diagonal(covariance, 1)

    return np.random.multivariate_normal(mean, covariance)


def get_m_prob(M, A=None):
    """
    Get probability of x from prior Dirichlet PDF on mineral abundance:
    1/B(a) * prod m_i^(a_i-1)
    :param M: vector of mineral abundances, sum to 1
    """
    def get_B(A):
        """
        Get multinomial beta function value given vector A of concentration parameters
        """
        numerator = 1
        for a in A:
            numerator *= math.gamma(a)
        denominator = math.gamma(np.sum(A))
        return numerator / denominator
    if A is None:
        A = np.array([1] * len(M))
    f = 1 / get_B(A)
    running_prod = 1
    for index, m in enumerate(M):
        running_prod *= m**(A[index] - 1)

    return f * running_prod


def get_D_prob(X):
    """
    Get probability of x from prior PDF on grain size:
    p(x) = 1 / (b-a)
    :param X: vector grain size
    """
    min_grain_size = 25
    max_grain_size = 50
    D_probs = []
    for x in X:
        D_probs.append(1 / (max_grain_size - min_grain_size))
    return np.array(D_probs)


def sample_likelihood(m, D):
    """
    Get p(d|m, D) through ancestral sampling
    """
    r_e = get_w_mixed_hapke_estimate(m, D)

    r = get_reflectance_spectra(sid, sample_spectra)

    stats.multivariate_normal(mean, covariance)

    covariance = np.zeros((length, length))
    np.fill_diagonal(covariance, 5 * 10 ^ (-4))
    y = stats.multivariate_normal.pdf(r, mean=r_e, cov=covariance)

    return y


def transition_params(cur_m, cur_D):
    """
    Determine whether or not to accept the new parameters, based on the ratio of likelihood*priors
    :param cur_m: Vector of mineral abundances
    :param cur_D: Vector of grain sizes
    """
    new_m, new_D = sample_params(cur_m, cur_D)

    m_prior = get_m_prob(cur_m)
    D_prior = get_D_prob(cur_D)


def sample_params(cur_m, cur_D):
    """
    Sample new m and D
    """
    new_m = sample_dirichlet(cur_m)
    new_D = sample_multivariate(cur_D)
    return new_m, new_D
