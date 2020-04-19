import numpy as np
import imageio
from skimage.draw import line_aa, circle, line
from tqdm import tqdm
from heapq import heapify, heappop, heappush
from node import Node
from utils import xy2rc, rc2xy, bold_filter, array_in_list, distance
import copy


def read_map_to_nparray(uri):
    map_img = imageio.imread(uri)
    return np.array(map_img)


class PRM:
    def __init__(self, waypoints, sim_map, n_vertices=10, vertex_value=20):
        self.waypoints_xy = waypoints  # list of numpy (2,) np array representing coordinates
        # this is a dictionary of vertices. The key is the coordinate in x,y convention
        # The values are each vertices in a circle with a desired radius
        self.vertices_xy = {}  # in xy convention
        self.graph_xy = {}
        self.sim_map = sim_map
        # This was to create map that accounts for the "thickness" of the robot
        # This way, the simulation can be performed assuming the robot is a particle
        self.sim_map_bolded = bold_filter(self.sim_map, 5)
        self.shortest_path_map = None
        self.shortest_path_stacks = None

        self.x_dim = self.sim_map.shape[1]
        self.y_dim = self.sim_map.shape[0]
        self.NUM_VERTICES = 4 + n_vertices
        self.vertex_value = vertex_value
        self.r_for_nearest_neighbour = 25
        self.radius = 3  # 0.45m to pixel thickness
        self.obstacles = np.argwhere(self.sim_map_bolded == 0)  # in rc convention

        # these were added to help go through the very narrow corridor
        helper_vertices_rc = [(18, 79), (86, 17)]

        # Adding the helper vertices into the vertices list
        for helper_vertex_rc in helper_vertices_rc:
            rr, cc = circle(helper_vertex_rc[0], helper_vertex_rc[1], self.radius)
            circles_coords_rc = np.vstack([rr, cc]).T
            self.vertices_xy[(helper_vertex_rc[1], helper_vertex_rc[0])] = circles_coords_rc

        # Adding the waypoints into the vertices list
        for view_point_xy in self.waypoints_xy:
            view_point_rc = xy2rc(view_point_xy)
            key = tuple(view_point_xy)
            rr, cc = circle(view_point_rc[0], view_point_rc[1], 2.5)
            circles_coords_rc = np.vstack([rr, cc]).T
            circles_coords_rc[circles_coords_rc > 95] = 95
            self.vertices_xy[key] = circles_coords_rc

        return

    def add_vertices(self):
        """
         Randomly samples vertices, and adds it to the vertex list if the sampled vertex is not on top of an obstacle
        :return: None
        """
        vertices_added = 0
        np.random.seed(seed=19940513)
        while vertices_added < self.NUM_VERTICES:
            x = int(np.random.uniform(0, self.x_dim))
            y = int(np.random.uniform(0, self.y_dim))
            p_xy = np.array([x, y])
            p_rc = xy2rc(p_xy)
            rr, cc = circle(p_rc[0], p_rc[1], self.radius)
            circles_coords_rc = np.vstack([rr, cc]).T
            circles_coords_rc[circles_coords_rc > 95] = 95
            valid_vertex = True
            if array_in_list(p_rc, list(self.obstacles)) or array_in_list(p_xy, self.vertices_xy):
                valid_vertex = False
            if valid_vertex:  # It is valid vertex if the sampled vertex is not on an obstacle
                print("{},{} -> add vertex".format(x, y))
                key = tuple(p_xy)
                self.vertices_xy[key] = circles_coords_rc
                vertices_added += 1
            else:
                pass

    def check_collision(self, p1xy, p2xy):
        """
        checks collision by checking if line drawn using Bresenham algorithm intersects with any obstacle
        :param p1xy: point 1 in xy coordinate
        :param p2xy: point 2 in xy coordinate
        :return: Boolean whether there is a collision present or not between the two points
        """
        p1rc = xy2rc(p1xy)
        p2rc = xy2rc(p2xy)

        rr, cc = line(int(p1rc[0]), int(p1rc[1]), int(p2rc[0]), int(p2rc[1]))
        line_coords_rc = np.vstack([rr, cc]).T
        for line_coord_rc in line_coords_rc:
            if array_in_list(line_coord_rc, list(self.obstacles)):
                return True
        return False

    def nearest_neighbours(self, q1xy):
        """
        Scans the area with specified radius, and finds any vertices that are within the area, that is not obstructed
        :param q1xy: Vertex of interest in xy coordinate
        :return: list of points that constitutes as neighbours
        """
        nn_xy = []
        for q2xy_key in self.vertices_xy.keys():
            q2xy = np.array(q2xy_key)

            if (q1xy != q2xy).all():
                dist = distance(q1xy, q2xy)
                if self.radius < dist < self.r_for_nearest_neighbour:
                    collisionPresent = self.check_collision(q1xy, q2xy)
                    if not collisionPresent:
                        nn_xy.append((q2xy, dist))
        return nn_xy

    def form_graph(self):
        """
        Forms graph based on the nearest neighbour output.
        Each vertex is a key, and the values are all the vertices that are connected to that vertex
        :return: None
        """
        for q_xy in tqdm(self.vertices_xy.keys()):
            nn_xy = self.nearest_neighbours(q_xy)
            key = tuple(q_xy)  # to make it hashable
            if key not in self.graph_xy:
                self.graph_xy[key] = nn_xy  # [(p0, dist0), (p1, dist1), (p2, dist2), (p3, dist3) ...]
            else:
                raise ValueError(f"SHOULD NOT BE HERE{key}")

    def find_shortest_paths(self):
        """
        Finds shortest path
        :return: None
        """
        stacks = []
        # Goes through the waypoint in pairs, and finds the shortest path between those two waypoints
        for start, end in zip(self.waypoints_xy[:-1], self.waypoints_xy[1:]):
            stack = self.a_star(start, end)
            stacks.append(stack)
            print(f"start:{start}, end:{end}, stack:{stack}")
        self.shortest_path_stacks = copy.deepcopy(stacks)
        self.visualize_shortest_paths(stacks)

    # A* algorithm implementation using priority queue
    def a_star(self, start_xy, end_xy):
        """
        A* Algorithm using priority queue
        :param start_xy: Strating point in xy coordinate
        :param end_xy: Ending point in xy coordinate
        :return: Stack that represents the shortest path
        """
        start_node = Node(start_xy, end_xy, self.graph_xy)
        start_node.shortest_dist = 0
        start_node.update_total_cost()
        pq = []
        pq.append(start_node)
        heapify(pq)
        stack = []
        while pq:
            current_node = heappop(pq)
            if (current_node.coord_xy == end_xy).all():
                print("it's done ")
                iterator = current_node
                while iterator:
                    stack.append(iterator.coord_xy)
                    iterator = iterator.prev_node
                break

            for neighbour, dist in current_node.connections:
                neighbouring_node = Node(neighbour, end_xy, self.graph_xy)
                if current_node.shortest_dist + dist < neighbouring_node.shortest_dist:
                    neighbouring_node.shortest_dist = current_node.shortest_dist + dist
                    neighbouring_node.update_total_cost()
                    neighbouring_node.prev_node = current_node
                    heappush(pq, neighbouring_node)
        return stack

    def visualize_connection(self):
        temp_map = np.copy(self.sim_map_bolded)
        for vertex_xy in self.graph_xy.keys():
            for connection_xy, weight in self.graph_xy[vertex_xy]:
                connection_rc = xy2rc(connection_xy)
                vertex_rc = xy2rc(vertex_xy)
                rr, cc, val = line_aa(vertex_rc[0], vertex_rc[1], connection_rc[0], connection_rc[1])
                temp_map[rr, cc] = 0
                circles_coord = self.vertices_xy[tuple(vertex_xy)]
                circles_coord = circles_coord.T
                rr_circle = circles_coord[0, :]
                cc_circle = circles_coord[1, :]
                temp_map[rr_circle, cc_circle] = 0

        r_ch = np.copy(temp_map)
        g_ch = np.copy(temp_map)
        all_connected = True
        for view_point_xy in self.waypoints_xy:
            if all_connected:
                all_connected = len(self.graph_xy[tuple(view_point_xy)]) > 0
            view_point_rc = xy2rc(view_point_xy)
            rr, cc = circle(view_point_rc[0], view_point_rc[1], self.radius)
            r_ch[rr, cc] = 255
            g_ch[rr, cc] = 0
            temp_map[rr, cc] = 0
        img = np.append(r_ch[..., np.newaxis], np.append(g_ch[..., np.newaxis], temp_map[..., np.newaxis], -1), axis=-1)
        self.shortest_path_map = img
        imageio.imwrite("plots/prm/connections.png", img)
        print(all_connected)

    def visualize_shortest_paths(self, paths):
        for stack in paths:
            start_node = stack.pop()
            start_node_rc = xy2rc(start_node)
            while stack:
                next_node = stack.pop()
                print(f"start_node:{start_node}, next_node:{next_node}")
                next_node_rc = xy2rc(next_node)
                rr, cc = line(start_node_rc[0], start_node_rc[1], next_node_rc[0], next_node_rc[1])
                self.shortest_path_map[rr, cc, 0] = 0
                self.shortest_path_map[rr, cc, 1] = 255
                self.shortest_path_map[rr, cc, -1] = 0
                start_node_rc = next_node_rc

        imageio.imwrite("plots/prm/shortest_paths.png", self.shortest_path_map)


def main():
    sim_map = read_map_to_nparray('images/map.png')
    # The waypoints that must be visited
    view_points_xy = [np.array([70, 15]), np.array([90, 50]), np.array([30, 95]), np.array([5, 50]), np.array([5, 5])]
    prm = PRM(view_points_xy, sim_map)
    prm.add_vertices()
    prm.form_graph()
    prm.visualize_connection()
    prm.find_shortest_paths()

    for view_point_xy in view_points_xy:
        view_point_rc = xy2rc(view_point_xy)
        rr, cc = circle(view_point_rc[0], view_point_rc[1], 2.5)
        sim_map[rr, cc] = 0


if __name__ == "__main__":
    main()
