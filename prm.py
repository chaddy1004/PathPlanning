import numpy as np
from read_pgm import read_pgm
import imageio
from skimage.draw import line_aa, circle, line
from tqdm import tqdm
from heapq import heapify, heappop, heappush
from node import Node
from utils import xy2rc, rc2xy, bold_filter, array_in_list, distance
from robot import Robot
import copy


class PRM:
    def __init__(self, waypoints, sim_map, n_vertices=10, vertex_value=20):
        self.waypoints_xy = waypoints  # list of numpy (2,) np array representing coordinates
        self.vertices_xy = {}  # in xy convention
        self.graph_xy = {}
        self.sim_map = sim_map
        self.sim_map_bolded = bold_filter(self.sim_map, 5)
        self.traverse_map = np.copy(sim_map)
        self.shortest_path_map = None
        self.shortest_path_stacks = None

        self.x_dim = self.sim_map.shape[1]
        self.y_dim = self.sim_map.shape[0]
        self.NUM_VERTICES = 4 + n_vertices
        self.vertex_value = vertex_value
        self.r = 25
        self.radius = 3  # 0.45m to pixel thickness
        self.obstacles = np.argwhere(self.sim_map_bolded == 0)  # in rc convention

        self.robot = Robot(np.array([70, 15, np.deg2rad(45)]), dt=0.1)

        helper_vertices_rc = [(18, 79), (
            86, 17)]  # , (18, 82), (18, 83)] #(15, 84), (15, 85), (15, 86), (15, 87), (15,88), (15,89), (15,90)]
        for helper_vertex_rc in helper_vertices_rc:
            rr, cc = circle(helper_vertex_rc[0], helper_vertex_rc[1], self.radius)
            circles_coords_rc = np.vstack([rr, cc]).T
            self.vertices_xy[(helper_vertex_rc[1], helper_vertex_rc[0])] = circles_coords_rc

        for view_point_xy in self.waypoints_xy:
            view_point_rc = xy2rc(view_point_xy)
            key = tuple(view_point_xy)
            rr, cc = circle(view_point_rc[0], view_point_rc[1], 2.5)
            self.traverse_map[rr,cc] = 0
            circles_coords_rc = np.vstack([rr, cc]).T
            circles_coords_rc[circles_coords_rc > 95] = 95
            self.vertices_xy[key] = circles_coords_rc

        return

    def add_vertices(self):
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
            if valid_vertex:
                print("{},{} -> add vertex".format(x, y))
                key = tuple(p_xy)
                # rr_v, cc_v = circle(p_rc[0], p_rc[1], self.radius)
                self.vertices_xy[key] = circles_coords_rc
                # self.sim_map[rr_v, cc_v] = 0
                vertices_added += 1
            else:
                pass

    def check_collision(self, p1xy, p2xy):
        p1rc = xy2rc(p1xy)
        p2rc = xy2rc(p2xy)

        rr, cc = line(int(p1rc[0]), int(p1rc[1]), int(p2rc[0]), int(p2rc[1]))
        line_coords_rc = np.vstack([rr, cc]).T
        for line_coord_rc in line_coords_rc:
            if array_in_list(line_coord_rc, list(self.obstacles)):
                return True
        return False

    def nearest_neighbours(self, q1xy):
        nn_xy = []
        for q2xy_key in self.vertices_xy.keys():
            q2xy = np.array(q2xy_key)

            if (q1xy != q2xy).all():
                dist = distance(q1xy, q2xy)
                if self.radius < dist < self.r:
                    collisionPresent = self.check_collision(q1xy, q2xy)
                    if not collisionPresent:
                        nn_xy.append((q2xy, dist))
        return nn_xy

    def form_graph(self):
        for q_xy in tqdm(self.vertices_xy.keys()):
            nn_xy = self.nearest_neighbours(q_xy)
            key = tuple(q_xy)  # to make it hashable
            if key not in self.graph_xy:
                self.graph_xy[key] = nn_xy  # [(p0, dist0), (p1, dist1), (p2, dist2), (p3, dist3) ...]
            else:
                raise ValueError(f"SHOULD NOT BE HERE{key}")

    def visualize_connection(self):
        temp_map = np.copy(self.sim_map_bolded)
        # temp_map = self.sim_map_bolded
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

    def find_shortest_paths(self):
        stacks = []
        for start, end in zip(self.waypoints_xy[:-1], self.waypoints_xy[1:]):
            stack = self.a_star(start, end)
            stacks.append(stack)
            print(f"start:{start}, end:{end}, stack:{stack}")
        self.shortest_path_stacks = copy.deepcopy(stacks)
        self.visualize_shortest_paths(stacks)

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

    def a_star(self, start_xy, end_xy):
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
                print("it';s done ")
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

    def _traverse(self, path, i):
        path = path[::-1]
        temp_map = np.copy(self.traverse_map)
        for start, end in zip(path[:-1], path[1:]):
            direction = end - start
            self.robot.set_heading_angle(direction)
            self.robot.rotate_to_heading_angle()
            at_destination = False
            while not at_destination:
                self.robot.move_straight_xy(0.1)
                current_pos_xy = np.array([self.robot.state[0], self.robot.state[1]])
                error = np.linalg.norm(end-current_pos_xy)
                print(error, current_pos_xy, end)
                # current_pos_xy = np.array([int(self.robot.state[0]), int(self.robot.state[1])])
                if np.linalg.norm(end-current_pos_xy) < 3:
                    at_destination = True
                current_pos_rc = xy2rc(current_pos_xy)
                self.traverse_map[int(current_pos_rc[0]), int(current_pos_rc[1])] = 0
                temp_map[int(current_pos_rc[0]), int(current_pos_rc[1])] = 0
        imageio.imwrite(f"plots/prm/{i}_robot_traversal.png", self.traverse_map)

        return

    def traverse(self):
        for i, path in enumerate(self.shortest_path_stacks):
            self._traverse(path, i)
        imageio.imwrite("plots/prm/robot_traversal.png", self.traverse_map)
        return


def main():
    sim_map = read_pgm('sim_map.pgm')
    sim_map = np.array(sim_map)
    view_points_xy = [np.array([70, 15]), np.array([90, 50]), np.array([30, 95]), np.array([5, 50]), np.array([5, 5])]
    #
    prm = PRM(view_points_xy, sim_map)
    prm.add_vertices()
    prm.form_graph()
    prm.visualize_connection()
    prm.find_shortest_paths()
    prm.traverse()

    for view_point_xy in view_points_xy:
        view_point_rc = xy2rc(view_point_xy)
        rr, cc = circle(view_point_rc[0], view_point_rc[1], 2.5)
        sim_map[rr, cc] = 0


if __name__ == "__main__":
    main()
