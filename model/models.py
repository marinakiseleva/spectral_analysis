import numpy as np
import multiprocessing

from model.inference import *
from model.mrf_inference import MRFModel
from model.segmentation import segment_image, get_superpixels
from utils.plotting import *
from utils.constants import *


def mrf_model(iterations, image, V, C):
    """
    MRF model, with single setting for beta
    """

    pool = multiprocessing.Pool(NUM_CPUS)

    model = MRFModel(beta = 10, 
                    iterations=iterations,
                    image=image,
                    V=V,
                    C=C)

    # func = partial(infer_mrf_image, iterations=iterations,
    #                                image=image,
    #                                V=V,
    #                                C=C)
 

    # betas = [0.1, 1, 10, 100]
    # mDs = []
    # # Multithread over the pixels' reflectances
    # mDs = pool.map(func, betas)
    # pool.close()
    # pool.join()

    # energies = []
    # for i, mD in enumerate(mDs):
    #     energy = get_total_energy(image, mD[0], mD[1], betas[i])

    #     energies.append(energy)
    # min_i = energies.index(min(energies))

    # print("\n\noptimal beta="+ str(betas[min_i]))    

    # return mDs[min_i]
    return model.infer_mrf_image()


def ind_model(iterations, image, V, C):
    """
    Use pixel-independent model to infer mineral assemblages and grain sizes of pixels in image
    """
    m_est, D_est = infer_image(iterations=iterations,
                               image=image,
                               V=V,
                               C=C)
    
    return m_est, D_est


def seg_model(seg_iterations, iterations, image, V, C, MAX_SAD):
    """
    Use segmentation model to infer mineral assemblages and grain sizes of pixels in image
    """

    graphs = segment_image(iterations=seg_iterations,
                           image=image,
                           MAX_SAD=MAX_SAD)
    superpixels = get_superpixels(graphs)
    print("Number of superpixels: " + str(len(superpixels)))

    m_and_Ds = infer_superpixels(iterations=iterations,
                                 superpixels=superpixels,
                                 V=V,
                                 C=C)

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
