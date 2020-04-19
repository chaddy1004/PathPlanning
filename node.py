import numpy as np
from utils import distance

# the node class used to form a graph for performing A*
class Node:
    def __init__(self, coord_xy, end_xy, graph):
        self.key = tuple(coord_xy)
        self.coord_xy = coord_xy
        self.heuristic = distance(self.coord_xy, end_xy)
        self.shortest_dist = np.inf
        self.prev_node = None
        self.total_cost = 0
        self.connections = graph[self.key]

    def __lt__(self, other):
        return self.total_cost < other.total_cost

    def update_total_cost(self):
        self.total_cost = self.heuristic + self.shortest_dist
