"""
Generates data by the following steps:

1. Create grid-based image with X different regions of unique mineralogy & label each region numerically
2. Create X distinct sets of minerals (different mixtures)
3. Generate image by assigning mixtures to defined regions

"""

import numpy as np
import math

from hapke_model import get_r_mixed_hapke_estimate
from constants import pure_endmembers, c_wavelengths
from plotting import *


def create_labeled_grid(X, categories, size=200):
    """
    Create grid (Numpy 2D array), with each quadrant corresponding to one of X numeric labels
    :param X: Grid dimensions, XxX grid squares
    :param categories: Number of categories, 1 up to this number will be randomly generated for each grid.
    """
    image = np.ones((size, size))
    quadrant_size = math.floor(size / X)
    i_start_index = 0
    for i in range(X):
        i_end_index = i_start_index + quadrant_size
        j_start_index = 0
        for j in range(X):
            j_end_index = j_start_index + quadrant_size
            if j_end_index >= size:
                j_end_index = size - 1
            if i_end_index >= size:
                i_end_index = size - 1

            a = np.random.randint(1, categories + 1)
            image[i_start_index:i_end_index + 1, j_start_index:j_end_index + 1] = a

            j_start_index = j_end_index
        i_start_index = i_end_index
    return image


class Mixture:

    def __init__(self, m, D, r):
        """
        :param m: vector
        :param D: vector
        """
        self.m = m
        self.D = D
        self.r = r


def create_mixture():
    """
    Create random mixture of 3 endmembers
    :return mixture: as Numpy array
    """
    m_random = np.random.dirichlet(np.ones(3), size=1)[0]
    D_random = np.random.randint(low=45, high=76, size=3)

    m_map = {}
    D_map = {}
    for index, endmember in enumerate(pure_endmembers):
        m_map[endmember] = m_random[index]
        D_map[endmember] = D_random[index]
    r = get_r_mixed_hapke_estimate(m_map, D_map)
    return Mixture(list(m_map.values()),
                   list(D_map.values()),
                   r)


class HSImage:

    def __init__(self, m_image, D_image, r_image, labeled_image):
        """
        Save different types of images for HSI
        """
        self.m_image = m_image
        self.D_image = D_image
        self.r_image = r_image
        self.labeled_image = labeled_image


def generate_image(num_mixtures, num_regions, noise_scale=0.01, size=200):
    """
    Creates images with num_mixtures random mixtures, in roughly num_regions^2
    :param num_regions: num_regions X num_regions regions 
    :param num_mixtures: Number of mixtures
    """
    labeled_image = create_labeled_grid(num_regions, num_mixtures, size)

    # Reflectance image
    r_image = np.ones((size, size, len(c_wavelengths)))
    # Mineral assemblage image
    m_image = np.ones((size, size, 3))
    # Grain size image
    D_image = np.ones((size, size, 3))

    mixtures_r = {}
    mixtures_m = {}
    mixtures_D = {}
    for i in range(num_mixtures):
        mix = create_mixture()
        mixtures_r[i + 1] = mix.r
        mixtures_m[i + 1] = mix.m
        mixtures_D[i + 1] = mix.D

    for i in range(size):
        for j in range(size):
            mix_i = int(labeled_image[i, j])
            r = mixtures_r[mix_i]
            # Add noise to reflectance
            noise = np.random.normal(loc=0, scale=noise_scale, size=len(c_wavelengths))

            r_image[i, j] = r + noise
            m_image[i, j] = mixtures_m[mix_i]
            D_image[i, j] = mixtures_D[mix_i]

    image = HSImage(m_image, D_image, r_image, labeled_image)
    return image

import matplotlib
if __name__ == "__main__":

    image = generate_image(5, 5, noise_scale=0.001, size=10)
    rgb_image = image.r_image[:, :, 1:4]
    plt.imshow(rgb_image, interpolation='nearest')
    plt.title("RGB of spectral data")
    plt.show()

    plt.imshow(image.labeled_image, interpolation='nearest')
    plt.title("Labeled")
    plt.show()
