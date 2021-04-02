"""
Segment image, by merging when:

    get_smallest_common_weight(A, B) <= get_min_max_weight(A,B)

where get_smallest_common_weight(A,B) is the smallest intersegment distance between 2 segments A and B:

    get_smallest_common_weight(A,B) = min _(i,j) (d(i,j))

and get_min_max_weight(A,B) is the minimum of the greatest weights in each segment

    get_min_max_weight(A,B) = min(GMW(A) + k/|A|, GMW(B) + k/|B|)

where GMW is the maximum edge weight in that minimum spanning tree, and |.| represents length of vector. Distance d, is calculated using SAD:

              (i^T * j)
    d(i,j) = -----------
             ||i|| ||j||

"""
import numpy as np
from functools import reduce
from utils.constants import *


"""
MAX_SAD is the initial threshold by which to merge pixels when both segments do not have edges to compare. If their SAD is <MAX_SAD we merge them. Comes from manually examining pure data spectra vs. derived Hapke model spectra where dominant mineral is 80%. 

"""


class Graph:

    def __init__(self, vertices, edges):
        """
        :param vertices: List of Vertex objects
        :param edge: List of Edge objects
        """
        if type(vertices) != list:
            raise ValueError("Vertices must be list.")
        if type(edges) != list:
            raise ValueError("Edges must be list.")
        self.vertices = vertices
        self.edges = edges


class Vertex:

    def __init__(self, value, neighbors, x, y):
        """
        :param value: reflectance (Numpy array)
        :param neighbors: list of Vertex neighbors (Numpy array)
        :param x: x index in image
        :param y: y index in image
        """
        if type(neighbors) != list:
            raise ValueError("neighbors must be list.")

        self.value = value
        self.neighbors = neighbors
        self.x = x
        self.y = y


class Edge:

    def __init__(self, a, b):
        """
        :param a: Vertex a
        :param b: Vertex b
        """
        if type(a) != Vertex:
            raise ValueError("a must be Vertex.")
        if type(b) != Vertex:
            raise ValueError("b must be Vertex.")
        self.a = a
        self.b = b
        self.value = get_SAD(a.value, b.value)

import random


def get_superpixels(graphs):
    """
    Get superpixel for each graph (average reflectance of vertices in that graph)
    """
    superpixels = []
    for index, g in enumerate(graphs):
        clus_num = index + 1
        reflectances = []

        for v in g.vertices:
            reflectances.append(v.value)

        avg_reflectance = np.average(reflectances, axis=0)
        superpixels.append(avg_reflectance)

    return superpixels


def segment_image(iterations, image):
    """
    Segments image using agglomerative clustering. At each iteration, randomly selects graph pair to potentially merge.

    """
    graphs = init_graphs(image)
    prev_clusters = []
    for i in range(iterations):
        if len(graphs) == 0:
            raise Error("Graph has been reduced to 1 cluster.")
        g1 = random.choice(graphs)
        g2 = random.choice(graphs)
        if g1 == g2:
            continue
        if can_merge(g1, g2):
            g3 = merge(g1, g2)
            graphs.remove(g1)
            graphs.remove(g2)
            graphs.append(g3)

        num_clusters = len(graphs)
        if len(prev_clusters) >= SEG_EARLY_STOP:
            prev_clusters.pop(0)
        prev_clusters.append(num_clusters)

        if i > SEG_BURN_IN and len(prev_clusters) == SEG_EARLY_STOP and prev_clusters.count(num_clusters) == len(prev_clusters):
            print("\nEarly Stopping in Segmentation, iter " + str(i))
            break

    """
     Merge any like superpixels/graphs
    Note: this is necessary because two separate clusters may be unconnected but may be of the same superpixel type.
    """
    print("Original number of superpixels: " + str(len(graphs)))
    superpixels = get_superpixels(graphs)
    skip_indices = []
    new_graphs = []
    for index, cur_superpixel in enumerate(superpixels):
        if index in skip_indices:
            continue
        cur_graph = graphs[index]
        for index2, compare_superpixel in enumerate(superpixels):
            compare_graph = graphs[index2]
            if index == index2:
                continue
            # merge superpixels if the SAD of their average reflectances < MAX_MERGE_SAD
            sad = get_SAD(cur_superpixel, compare_superpixel)
            if sad < MAX_MERGE_SAD:
                cur_graph = merge(cur_graph, compare_graph)
                skip_indices.append(index)
                skip_indices.append(index2)
        new_graphs.append(cur_graph)
    print("# of merged superpixels: " + str(len(new_graphs)))
    return new_graphs


def visualize_clusters(graphs, image):
    """
    Visualize clusters by saving unique cluster assignments
    """
    num_clusters = len(graphs)
    print(str(num_clusters) + " clusters total.")

    num_rows = image.shape[0]
    num_cols = image.shape[1]
    cluster_image = np.zeros((num_rows, num_cols))
    for index, g in enumerate(graphs):
        clus_num = index + 1
        for v in g.vertices:
            cluster_image[v.x, v.y] = clus_num
    print(cluster_image.shape)
    return cluster_image


def init_graphs(image):
    """
    Each pixel is its own singleton graph
    """
    num_rows = image.shape[0]
    num_cols = image.shape[1]
    # Initially save all verties in graph
    vertices = [[None] * num_cols for _ in range(num_rows)]
    for x in range(num_rows):
        for y in range(num_cols):
            v = Vertex(value=image[x, y],
                       neighbors=[],
                       x=x,
                       y=y)
            vertices[x][y] = v

    # Flatten vertices to 1d list
    vertices_flatten = reduce(lambda x, y: x + y, vertices)

    graphs = []
    # Update all vertices with neighbors
    for v in vertices_flatten:
        nbrs_list = get_neighbors(v, vertices, num_rows, num_cols)
        v.neighbors = nbrs_list
        graphs.append(Graph([v], []))
    return graphs


def create_edge(a, b, edges):
    """
    Create edge between a and b if it does not already exist in edges
     edge(a,b) = edge(b,a)
    :param a: Vertex a
    :param b: Vertex b
    """
    # candidate_edge = Edge(v, nbr_v)
    for known_edge in edges:
        if known_edge.a == a:
            if known_edge.b == b:
                return None
        if known_edge.a == b:
            if known_edge.b == a:
                return None
    # There is no duplicates - create edge
    return Edge(a, b)


def get_neighbors(v, vertices, num_rows, num_cols):
    """
    Get neighbor Vertices of vertex v
    """
    nbrs = []
    y = v.y
    x = v.x
    # Neighbors above
    if x != 0:
        # Top neighbor
        nbrs.append(vertices[x - 1][y])

        # Top right
        if y != num_cols - 1:
            nbrs.append(vertices[x - 1][y + 1])

        # Top left
        if y != 0:
            nbrs.append(vertices[x - 1][y - 1])

    # Neighbors below
    if x != num_rows - 1:
        # Bottom neighbor
        nbrs.append(vertices[x + 1][y])

        # Bottom right
        if y != num_cols - 1:
            nbrs.append(vertices[x + 1][y + 1])

        # Bottom left
        if y != 0:
            nbrs.append(vertices[x + 1][y - 1])

    # Right neighbor
    if y != num_cols - 1:
        nbrs.append(vertices[x][y + 1])

    # Left neighbor
    if y != 0:
        nbrs.append(vertices[x][y - 1])
    return nbrs


def merge(g1, g2):
    """
    Merge graphs with smallest edge weight (so that the connection is between the most spectrally similar pixels)
    """
    edges = get_connecting_edges(g1, g2)
    min_edge_weight = 1
    min_edge = None
    for edge in edges:
        if edge.value < min_edge_weight:
            min_edge_weight = edge.value
            min_edge = edge

    combined_vertices = g1.vertices + g2.vertices
    combined_edges = g1.edges + g2.edges + [min_edge]
    new_graph = Graph(combined_vertices, combined_edges)
    return new_graph


def can_merge(g1, g2):
    """
    Boolean on whether to merge graphs. Returns True when:
        get_smallest_common_weight(A, B) <= get_min_max_weight(A,B)
    """
    edges = get_connecting_edges(g1, g2)
    if len(edges) == 0:
        return False
    if len(g1.edges) == 0 or len(g2.edges) == 0:
        # When graph has no edges, we need to use an initial baseline
        for edge in edges:
            if edge.value < MAX_SAD:
                return True
    scew = get_smallest_common_weight(edges)
    mmew = get_min_max_weight(g1, g2, k=0.001)
    return scew <= mmew


def get_smallest_common_weight(edges):
    """
    Get the smallest edge weight in list of edges
    """
    min_edge_weight = 1
    for edge in edges:
        min_edge_weight = min(edge.value, min_edge_weight)
    return min_edge_weight


def get_connecting_edges(g1, g2):
    """
    Get list of edges that connect g1 to g2
    """
    edges = []
    for vertex in g1.vertices:
        for neighbor in vertex.neighbors:
            if neighbor in g2.vertices:
                temp_edge = Edge(vertex, neighbor)
                edges.append(temp_edge)
    return edges


def get_min_max_weight(g1, g2, k=0.001):
    """
    Gets the maximum edge weight in each graph, and returns the minimum between them.
        min(GMW(A) + k/|A|, GMW(B) + k/|B|)

    """
    area1 = len(g1.vertices)
    area2 = len(g2.vertices)
    return min(get_max_weight(g1) + k / area1,
               get_max_weight(g2) + k / area2)


def get_max_weight(g):
    """
    Get largest weight in this graph.
    :param g: Graph g
    """
    max_edge_weight = 0
    for edge in g.edges:
        max_edge_weight = max(edge.value, max_edge_weight)
    return max_edge_weight


def get_SAD(a, b):
    """
    Get spectral angle distance:
        d(i,j) =  (i^T * j) / ( ||i|| ||j|| )
    :param a: Numpy vector
    :param b: Numpy vector
    """
    n = np.dot(a.transpose(), b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return np.arccos(n / d)
