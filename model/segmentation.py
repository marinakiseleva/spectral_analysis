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


class Graph:

    def __init__(self, edges, vertices):
        """
        :param edge: List of Edge objects
        :param vertices: List of Vertex objects
        """
        self.edges = edges
        self.vertices = vertices


class Vertex:

    def __init__(self, value, neighbors, x, y):
        """
        :param value: reflectance (Numpy array)
        :param neighbors: list of Vertex neighbors (Numpy array)
        :param x: x index in image
        :param y: y index in image
        """
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
        self.a = a
        self.b = b
        self.value = get_SAD(a.value, b.value)


def segment_image(image):
    """
    Segments image using agglomerative clustering. At each iteration, randomly selects graph to potentially merge.

    """
    graphs = init_graphs(image)
    return graphs


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
                       neighbors=None,
                       x=x,
                       y=y)
            vertices[x][y] = v
    # Update all vertices with neighbors
    # Flatten vertices to 1d list
    vertices_flatten = reduce(lambda x, y: x + y, vertices)

    edges = []
    for v in vertices_flatten:
        nbrs_list = get_neighbors(v, vertices, num_rows, num_cols)
        for nbr_v in nbrs_list:
            # Create Edge
            e = create_edge(v, nbr_v, edges)
            if e is not None:
                edges.append(e)
        v.neighbors = nbrs_list

    return Graph(edges, vertices_flatten)


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
    Boolean on whether to merge graphs. Returns True when:
        get_smallest_common_weight(A, B) <= get_min_max_weight(A,B)
    """
    scew = get_smallest_common_weight(g1, g2)
    mmew = get_min_max_weight(g1, g2, k=0.001)
    return scew <= mmew


def get_smallest_common_weight(g1, g2):
    """
    Get the smallest edge weight that connects Graph g1 to Graph g2
    """
    edges = get_connecting_edges(g1, g2)
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


def get_min_max_weight(g1, g2, k):
    """
    Gets the maximum edge weight in each graph, and returns the minimum between them.
        min(GMW(A) + k/|A|, GMW(B) + k/|B|)

    """
    length_1 = len(g1.vertices)
    length_2 = len(g2.vertices)
    return min(get_max_weight(g1) + k / length_1,
               get_max_weight(g2) + k / length_2)


def get_max_weight(g):
    """
    Get largest weight in this segment
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
        :param b:Numpy vector
    """
    n = np.dot(a.transpose(), b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return np.arccos(n / d)
