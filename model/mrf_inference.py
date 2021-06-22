"""
Runs  inference on the model to estimate the posterior p(m,D|d)
"""
import sys
from datetime import datetime
from functools import partial
import multiprocessing
import numpy as np
import math
from scipy.stats import multivariate_normal
from collections import OrderedDict

from model.inference import *
from model.hapke_model import get_USGS_r_mixed_hapke_estimate
from utils.constants import *


class MRFModel:
    def __init__(self, beta, iterations, image, V, C):
        """
        Initialize MRF model with hyperparams
        :param beta: smoothness parameter 
        :param iterations: Number of MCMC iterations to run for each datapoint
        :param image: 3D Numpy array with 3rd dimension equal to # of wavelengths
        :param V: covariance diagonal for grain size, D
        :param C: scaling factor for sampling mineral assemblage from Dirichlet, m
        """
        self.beta = beta
        self.iterations = iterations
        self.r_image = image
        self.V = V
        self.C = C

    def init_mrf(self):
        """
        Set random mineral & grain  assemblage for each pixel and return 3D Numpy array with 3rd dimension as assemblage
        """
        N = USGS_NUM_ENDMEMBERS

        num_rows = self.r_image.shape[0]
        num_cols = self.r_image.shape[1]
        m_image = np.zeros((num_rows, num_cols, N))
        D_image = np.zeros((num_rows, num_cols, N))
        for i in range(num_rows):
            for j in range(num_cols):
                reflectance = self.r_image[i, j]

                rand_D = D_transition(np.array([INITIAL_D] * N), self.V)
                rand_m = sample_dirichlet(np.array([float(1 / N)] * N), self.C)
                m_image[i, j] = rand_m
                D_image[i, j] = rand_D

        return m_image, D_image


    def get_spatial_energy(self, m_image, i, j, m):
        """
        Get spatial energy using default distance
        :param m_image: 3D Numpy array, mineral assemblages for pixels
        :param i: row index for datapoint
        :param j: col index for datapoint
        :param m: Mineral assemblage for pixel i,j to consider
        """

        num_rows = m_image.shape[0]
        num_cols = m_image.shape[1]

        e_spatial = 0
        cur_row = i
        cur_col = j
        row_above = i - 1
        row_below = i + 1
        left_col = j - 1
        right_col = j + 1

        # Sum over distance between pixel and each neighbor

        if row_above >= 0:
            # Above
            e_spatial += get_distance(m_image[row_above, j], m)
            if left_col >= 0:
                # Top left
                e_spatial += get_distance(m_image[row_above, left_col], m)
            if right_col < num_cols:
                # Top right
                e_spatial += get_distance(m_image[row_above, right_col], m)

        if row_below < num_rows:
            # Below
            e_spatial += get_distance(m_image[row_below, j], m)
            if left_col >= 0:
                # Bottom left
                e_spatial += get_distance(m_image[row_below, left_col], m)
            if right_col < num_cols:
                # Top right
                e_spatial += get_distance(m_image[row_below, right_col], m)

        if left_col >= 0:
            # Left
            e_spatial += get_distance(m_image[cur_row, left_col], m)
        if right_col < num_cols:
            # Right
            e_spatial += get_distance(m_image[cur_row, right_col], m)

        return e_spatial

    def get_total_energy(self, m_image, D_image):
        """
        Get the total MRF energy; we want this to decrease at each iteration
        :param m_image: 3D Numpy array, mineral assemblages for pixels
        :param D_image: 3D Numpy array, grain sizes for pixels
        """
        num_rows = m_image.shape[0]
        num_cols = m_image.shape[1]

        energy_sum = 0
        for x in range(num_rows):
            for y in range(num_cols):
                d = self.r_image[x, y]
                m = m_image[x, y]
                D = D_image[x, y]
                e_spatial = self.get_spatial_energy(m_image, x, y, m)
                e_spectral = -get_log_posterior_estimate(d, m, D)
                pixel_energy = e_spectral + (e_spatial * self.beta)
                energy_sum += (pixel_energy)

        return energy_sum

    def get_mrf_prob(self, m_image, D_image, i, j, m, D, d):
        """
        Get joint probability of this pixel i,j in image
        """
        # get energy of neighbors
        e_spatial = self.get_spatial_energy(m_image, i, j, m)
        # Do not use log - posterior; for some reason that results in never
        # rejecting candidates.
        p = get_posterior_estimate(d, m, D)
        # joint prob is likelihood - spatial energy
        return p - (e_spatial * self.beta)


    def infer_mrf_datapoint(self, m_image, D_image, i, j, d):
        """
        Run metropolis algorithm (MCMC) to estimate m and D using posterior
        Return m_image and D_image with updated values 
        :param m_image: 3D Numpy array, mineral assemblages for pixels
        :param D_image: 3D Numpy array, grain sizes for pixels
        :param i: row index for datapoint d
        :param j: col index for datapoint d
        :param d: data, 1 spectral sample (1D Numpy vector) 
        """
        cur_m = m_image[i, j]
        cur_D = D_image[i, j]
        new_m, new_D = transition_model(cur_m, cur_D, self.V, self.C) 

        cur = self.get_mrf_prob(m_image, D_image, i, j, cur_m, cur_D, d)
        new = self.get_mrf_prob(m_image, D_image, i, j, new_m, new_D, d)

        ratio = new / cur
        phi = min(1, ratio)
        u = np.random.uniform(0, 1)
        if phi >= u:
            cur_m = new_m
            cur_D = new_D

        m_image[i, j] = cur_m
        D_image[i, j] = cur_D
        return m_image, D_image



    def infer_mrf_image(self):
        """
        Infer m and D for entire image by minimizing:
        - log(P(y_i | x_i)) + sum_{n in neighbors} SAD(y_i, y_n) 
        1. Initialize random mineral assemblages for each pixel
        2. Loop over pixels for X iterations, and use MCMC to sample new assemblage for each pixel.
        """
        num_rows = self.r_image.shape[0]
        num_cols = self.r_image.shape[1] 
        m_image = np.ones((num_rows, num_cols, USGS_NUM_ENDMEMBERS)) 
        D_image = np.ones((num_rows, num_cols, USGS_NUM_ENDMEMBERS))

        print("Initialize pixels in image for beta=" + str(self.beta) + "... ") 
        m_image, D_image = self.init_mrf()

        rows = np.arange(0, num_rows)
        cols = np.arange(0, num_cols)

        prev_energy = 0
        prev_imgs = []  # save last MRF_PREV_STEPS imgs in case of early stopping
        energy_diffs = []
        MAP_mD = [m_image, D_image, 1000]
        for iteration in range(self.iterations):
            # Randomize order of rows and columns each iteration
            np.random.shuffle(cols)
            np.random.shuffle(rows)
            # Iterate over each pixel in image
            for i in rows:
                for j in cols:
                    d = self.r_image[i, j]
                    m_image, D_image = self.infer_mrf_datapoint(
                        m_image, D_image, i, j, d)

            # Print out iteration performance
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            prev_imgs.append([m_image, D_image])
            energy = self.get_total_energy(m_image, D_image)
            energy_diff = energy - prev_energy
            energy_diffs.append(energy_diff)
            # update MAP
            if energy < MAP_mD[2]:
                MAP_mD = [m_image, D_image, energy]

            prev_energy = energy  # reset prev energy
        
            ps = "Iteration " + str(iteration + 1) + "/" + str(self.iterations) 
            ps += " for beta=" + str(self.beta)
            ps += "; total MRF Energy: " + str(round(energy, 2))
            ps += "; energy change from last iteration (want negative): " + str(round(energy_diff, 2))
            print("\n"+ps)

            print("m image shape " + str(m_image.shape))

            sys.stdout.flush()


            # If average energy change last MRF_PREV_STEPS runs was less than
            # MRF_EARLY_STOP, stop
            if len(energy_diffs) > MRF_BURN_IN:
                a_e = np.average(energy_diffs[-MRF_PREV_STEPS:])
                if a_e > MRF_EARLY_STOP:
                    print("\nMRF Early Stop at iteration " +
                          str(iteration) + " with average energy " + str(a_e))
                    m_image, D_image = prev_imgs[-MRF_PREV_STEPS]
                    break


        return MAP_mD[:2] 
