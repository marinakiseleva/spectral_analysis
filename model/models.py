import numpy as np
from model.inference import *
from model.segmentation import segment_image, get_superpixels
from utils.plotting import *
from utils.constants import *


def mrf_model(iterations, image):
    """
    Use pixel-independent model to infer mineral assemblages and grain sizes of pixels in image
    """
    m_est, D_est = infer_mrf_image(iterations=iterations,
                                   image=image)

    return m_est, D_est


def ind_model(iterations, image, C, V):
    """
    Use pixel-independent model to infer mineral assemblages and grain sizes of pixels in image
    """
    m_est, D_est = infer_image(iterations=iterations,
                               image=image,
                               C=C,
                               V=V)
    
    return m_est, D_est


def seg_model(seg_iterations, iterations, image, C, V):
    """
    Use segmentation model to infer mineral assemblages and grain sizes of pixels in image
    """

    graphs = segment_image(iterations=seg_iterations,
                           image=image)
    superpixels = get_superpixels(graphs)
    print("Number of superpixels: " + str(len(superpixels)))

    m_and_Ds = infer_superpixels(iterations=iterations,
                                 superpixels=superpixels,
                                   C=C,
                                   V=V)

    # Reconstruct image
    num_rows = image.shape[0]
    num_cols = image.shape[1]
    # Mineral assemblage predictions
    m_est = np.ones((num_rows, num_cols, USGS_NUM_ENDMEMBERS))
    # Grain size predictions
    D_est = np.ones((num_rows, num_cols, USGS_NUM_ENDMEMBERS))
    for index, pair in enumerate(m_and_Ds):
        graph = graphs[index]
        for v in graph.vertices:
            # retrieve x, y coords
            # [i, j] = index_coords[index]
            m, D = pair
            m_est[v.x, v.y] = m
            D_est[v.x, v.y] = D
    return m_est, D_est
