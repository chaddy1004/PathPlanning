import numpy as np


def bold_filter(img, kernel=5):
    """
    Bolding image using max filter. This is used for obstacle.
    :param img: Image represented by numpy array
    :param kernel: Size of the bolding
    :return:
    """
    img_bolded = np.copy(img)
    row, col = img.shape
    for y in range(row - kernel):
        for x in range(col - kernel):
            min_value = np.min(img[y:y + kernel, x:x + kernel])
            img_bolded[y:y + kernel, x:x + kernel] = min_value
    return img_bolded


# To convert x-y coordinate to row-column coordinate
def xy2rc(p_xy):
    """
    Converts x-y coordinate to row-column coordinate
    :param p_xy: coordinate in xy
    :return: coordinate in rc
    """
    p_rc = np.zeros_like(p_xy)
    p_rc[0] = p_xy[1]
    p_rc[1] = p_xy[0]
    return p_rc


# To convert row-column coordinate to x-y coordinate
def rc2xy(p_rc):
    """
    Converst row-column coordinate to x-y coordinate
    :param p_rc: coordintae in rc
    :return: coordinate in xy
    """
    p_xy = np.zeros_like(p_rc)
    p_xy[0] = p_rc[1]
    p_rc[1] = p_rc[0]
    return p_xy


# Checks if an element (numpy ara
def array_in_list(element, list_arrays):
    """
    Checks if an elemet is in a list
    :param element: numpy array
    :param list_arrays: list of numpy arrays
    :return:
    """
    if len(list_arrays) == 0:
        return False
    for array in list_arrays:
        if (element == array).all():
            return True
    return False


def distance(p1, p2):
    """
    Euclidian distance
    :param p1: point 1
    :param p2: point 2
    :return: distance between two points
    """
    diff = p2 - p1
    return np.sqrt((diff[0] * diff[0]) + (diff[1] * diff[1]))


def are_graphs_connected(graph1, start1, graph2, start2):
    """
    Checks whether two graphs are connected or not using DFS
    :param graph1: First graph; dictionary
    :param start1: Starting node of graph 1
    :param graph2: Second graph; dictionary
    :param start2: Starting node of graph 2
    :return: Boolean whether if two graphs are connected
    """

    def dfs(graph, start):
        vertices = set()
        stack = []
        if type(start) == tuple:
            key = start
        else:  # if this is numpy array
            key = tuple(start)
        stack.append(key)
        vertices.add(key)
        while stack:
            current = stack.pop()
            if current not in vertices:
                for connection, _ in graph[current]:
                    key = tuple(connection)
                    stack.append(key)
        return vertices

    g1_vertices = dfs(graph1, start1)
    g2_vertices = dfs(graph2, start2)
    return g1_vertices == g2_vertices


def find_closest_node(graph, new_node):
    """
    Finds the closest node to a given new_node
    :param graph: Graph in the form of dictionary
    :param new_node: New node to be used as reference
    :return: node closest to the new_node (tuple), and dist (the distance between the node and new_node)
    """
    max_distance = np.inf
    closest_node = None
    for node in graph.keys():
        node_array = np.array(node)
        dist = distance(node_array, new_node)
        if dist < max_distance:
            max_distance = dist
            closest_node = node_array
    return closest_node, dist
