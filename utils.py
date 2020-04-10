import numpy as np


def bold_filter(img, kernel=5):
    """
    Bolds the image by iterating through all the pixels using a kernel and changes the pixels inside that kernel to the minimum (the darkest) value
    :param img: image in the form of numpy array. Assumed to have no channel, only width and height
    :param kernel: Size of kernel that iterates through the image
    """
    img_bolded = np.copy(img)
    row, col = img.shape
    for y in range(row - kernel):
        for x in range(col - kernel):
            min_value = np.min(img[y:y + kernel, x:x + kernel])
            img_bolded[y:y + kernel, x:x + kernel] = min_value
    return img_bolded


def xy2rc(p_xy):
    p_rc = np.zeros_like(p_xy)
    p_rc[0] = p_xy[1]
    p_rc[1] = p_xy[0]
    return p_rc


def rc2xy(p_rc):
    p_xy = np.zeros_like(p_rc)
    p_xy[0] = p_rc[1]
    p_rc[1] = p_rc[0]
    return p_xy


def array_in_list(element, list_arrays):
    if len(list_arrays) == 0:
        return False
    for array in list_arrays:
        if (element == array).all():
            return True
    return False


def distance(p1, p2):
    diff = p2 - p1
    return np.sqrt((diff[0] * diff[0]) + (diff[1] * diff[1]))


def are_graphs_connected(graph1, start1, graph2, start2):
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
    max_distance = np.inf
    closest_node = None
    for node in graph.keys():
        node_array = np.array(node)
        dist = distance(node_array, new_node)
        if dist < max_distance:
            max_distance = dist
            closest_node = node_array
    return closest_node, dist
