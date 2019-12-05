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

    return get_r_mixed_hapke_estimate(m_map, D_map)


def generate_image(num_mixtures, num_regions, size=200):
    """
    Creates images with num_mixtures random mixtures, in roughly num_regions^2
    :param num_regions: num_regions X num_regions regions 
    :param num_mixtures: Number of mixtures
    """
    image = create_labeled_grid(num_regions, num_mixtures, size)

    hs_image = np.ones((size, size, len(c_wavelengths)))

    mixtures = {}
    for i in range(num_mixtures):
        mixtures[i + 1] = create_mixture()

    for i in range(size):
        for j in range(size):
            r = mixtures[int(image[i, j])]
            hs_image[i, j] = r
    return hs_image


if __name__ == "__main__":
    hs_image = generate_image(5, 5, size=10)
