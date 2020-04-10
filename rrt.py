import numpy as np
from read_pgm import read_pgm
import imageio
from skimage.draw import line_aa, circle, line, circle_perimeter
from tqdm import tqdm
from node import Node
from utils import xy2rc, rc2xy, distance, array_in_list, bold_filter, are_graphs_connected, find_closest_node
from heapq import heappush, heappop, heapify
import copy
from robot import Robot

np.random.seed(19940513)


class RRT:
    def __init__(self, waypoints, sim_map):
        self.waypoints_xy = waypoints  # list of numpy (2,) np array representing coordinates
        self.graph_start_xy = {}
        self.graph_end_xy = {}
        self.sim_map = sim_map
        self.sim_map_bolded = bold_filter(self.sim_map, 5)
        self.traverse_map = np.copy(sim_map)
        self.shortest_path_map = None
        self.shortest_path_stacks = None

        self.x_dim = self.sim_map.shape[1]
        self.y_dim = self.sim_map.shape[0]
        self.r = 25
        self.radius = 3  # 0.45m to pixel thickness
        self.step = 15
        self.obstacles = np.argwhere(self.sim_map_bolded == 0)  # in rc convention

        self.robot = Robot(np.array([5, 5, np.deg2rad(45)]), dt=0.1)

        for view_point_xy in self.waypoints_xy:
            view_point_rc = xy2rc(view_point_xy)
            rr, cc = circle(view_point_rc[0], view_point_rc[1], 2.5)
            self.traverse_map[rr, cc] = 0
        return

    def check_collision(self, p1xy, p2xy):
        p1rc = xy2rc(p1xy)
        p2rc = xy2rc(p2xy)
        rr, cc = line(int(p1rc[0]), int(p1rc[1]), int(p2rc[0]), int(p2rc[1]))
        line_coords_rc = np.vstack([rr, cc]).T
        for line_coord_rc in line_coords_rc:
            if array_in_list(line_coord_rc, list(self.obstacles)):
                return True
        return False

    def rrt(self, graph_num, visualize=True):
        graphs_connected = False
        iteration = 0
        map = np.copy(self.sim_map_bolded)
        while not graphs_connected:
            graphs_connected = self._update_graph(main_graph=self.graph_start_xy,
                                                  target_graph=self.graph_end_xy,
                                                  radius=self.step)

            if visualize:
                self.visualize_rrt(graph=self.graph_start_xy, map=map,
                                   name=f"plots/rrt/{graph_num}/{iteration}_one.png")

            if not graphs_connected:
                graphs_connected = self._update_graph(main_graph=self.graph_end_xy,
                                                      target_graph=self.graph_start_xy,
                                                      radius=self.step)
                if graphs_connected:
                    self.graph_start_xy = self.graph_end_xy

            if visualize:
                self.visualize_rrt(graph=self.graph_end_xy, map=map, name=f"plots/rrt/{graph_num}/{iteration}_two.png")
            iteration += 1

        if visualize:
            self.visualize_rrt(graph=self.graph_start_xy, map=map,
                               name=f"plots/rrt/{graph_num}/{iteration + 1}_one.png")
            self.visualize_rrt(graph=self.graph_end_xy, map=map, name=f"plots/rrt/{graph_num}/{iteration + 1}_two.png")
        return True

    def _update_graph(self, main_graph, target_graph, radius):
        graph_connected = False
        new_node_xy = [np.random.randint(0, self.x_dim), np.random.randint(0, self.y_dim)]
        closest_node_xy, dist = find_closest_node(main_graph, new_node_xy)
        diff = new_node_xy - closest_node_xy
        dir_vec = diff / np.linalg.norm(diff)
        dir_vec = dir_vec * radius
        new_node_xy = closest_node_xy + dir_vec
        new_node_xy = new_node_xy.astype(int)
        if not self.check_collision(new_node_xy, closest_node_xy):
            main_graph[tuple(closest_node_xy)].append((new_node_xy, dist))
            main_graph[tuple(new_node_xy)] = []
            main_graph[tuple(new_node_xy)].append((closest_node_xy, dist))

        closest_node_from_target_xy, dist_2 = find_closest_node(target_graph, new_node_xy)
        if dist_2 < radius:
            if not self.check_collision(new_node_xy, closest_node_from_target_xy):
                target_graph[tuple(closest_node_from_target_xy)].append((new_node_xy, dist_2))
                target_graph[tuple(new_node_xy)] = []
                target_graph[tuple(new_node_xy)].append((closest_node_from_target_xy, dist_2))
                stack = []
                stack.append((closest_node_from_target_xy, dist_2))
                visited = set()
                while stack:
                    print(stack)
                    coord, dist = stack.pop()
                    key = tuple(coord)
                    visited.add(key)
                    connections = target_graph[key]
                    if key in main_graph:
                        main_graph[key] = main_graph[key] + connections
                    else:
                        main_graph[key] = connections
                    for node in connections:
                        if (tuple(node[0])) not in visited:
                            stack.append(node)

                graph_connected = True

        return graph_connected

    def find_shortest_paths(self):
        iteration = 0
        stacks = []
        for start, end in zip(self.waypoints_xy[:-1], self.waypoints_xy[1:]):
            print(start, end)
            start_key = tuple(start)
            end_key = tuple(end)
            self.graph_start_xy = {}
            self.graph_end_xy = {}
            self.graph_start_xy[start_key] = []
            self.graph_end_xy[end_key] = []
            self.rrt(iteration, visualize=False)
            stack = self.a_star(start, end)
            stacks.append(stack)
            self.visualize_connection(iteration)
            iteration += 1
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
        imageio.imwrite("plots/rrt/shortest_paths.png", self.shortest_path_map)

    def visualize_connection(self, iteration):
        temp_map = np.copy(self.sim_map_bolded)
        for vertex_xy in tqdm(self.graph_start_xy.keys()):
            for connection_xy, weight in self.graph_start_xy[vertex_xy]:
                connection_rc = xy2rc(connection_xy)
                vertex_rc = xy2rc(vertex_xy)
                rr, cc, val = line_aa(vertex_rc[0], vertex_rc[1], connection_rc[0], connection_rc[1])
                temp_map[rr, cc] = 0

        r_ch = np.copy(temp_map)
        g_ch = np.copy(temp_map)
        for view_point_xy in self.waypoints_xy:
            view_point_rc = xy2rc(view_point_xy)
            rr, cc = circle(view_point_rc[0], view_point_rc[1], self.radius)
            g_ch[rr, cc] = 0
            temp_map[rr, cc] = 0
            r_ch[rr, cc] = 255
        img = np.append(r_ch[..., np.newaxis], np.append(g_ch[..., np.newaxis], temp_map[..., np.newaxis], -1), axis=-1)
        self.shortest_path_map = img
        imageio.imwrite(f"plots/rrt/connections{iteration}.png", img)

    def visualize_rrt(self, graph, map, name):
        for vertex_xy in tqdm(graph.keys()):
            for connection_xy, weight in graph[vertex_xy]:
                connection_rc = xy2rc(connection_xy)
                vertex_rc = xy2rc(vertex_xy)
                rr, cc, val = line_aa(vertex_rc[0], vertex_rc[1], connection_rc[0], connection_rc[1])
                map[rr, cc] = 0
                rr, cc = circle(vertex_rc[0], vertex_rc[1], self.radius)
                rr[rr > 95] = 95
                cc[cc > 95] = 95
                map[rr, cc] = 0

        r_ch = np.copy(map)
        g_ch = np.copy(map)
        for view_point_xy in self.waypoints_xy:
            view_point_rc = xy2rc(view_point_xy)
            rr, cc = circle(view_point_rc[0], view_point_rc[1], self.radius)
            g_ch[rr, cc] = 0
            map[rr, cc] = 0
            r_ch[rr, cc] = 255
        img = np.append(r_ch[..., np.newaxis], np.append(g_ch[..., np.newaxis], map[..., np.newaxis], -1), axis=-1)
        imageio.imwrite(name, img)

    def a_star(self, start_xy, end_xy):
        start_node = Node(start_xy, end_xy, self.graph_start_xy)
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
                neighbouring_node = Node(neighbour, end_xy, self.graph_start_xy)
                if current_node.shortest_dist + dist < neighbouring_node.shortest_dist:
                    neighbouring_node.shortest_dist = current_node.shortest_dist + dist
                    neighbouring_node.update_total_cost()
                    neighbouring_node.prev_node = current_node
                    heappush(pq, neighbouring_node)
        return stack

    def _traverse(self, path):
        path = path[::-1]
        for start, end in zip(path[:-1], path[1:]):
            direction = end - start
            self.robot.set_heading_angle(direction)
            self.robot.rotate_to_heading_angle()
            at_destination = False
            while not at_destination:
                self.robot.move_straight_xy(0.1)
                current_pos_xy = np.array([self.robot.state[0], self.robot.state[1]])
                error = np.linalg.norm(end - current_pos_xy)
                print(error, current_pos_xy, end)
                if np.linalg.norm(end - current_pos_xy) < 5:
                    at_destination = True
                current_pos_rc = xy2rc(current_pos_xy)
                self.traverse_map[int(current_pos_rc[0]), int(current_pos_rc[1])] = 0
        return

    def traverse(self):
        for path in self.shortest_path_stacks:
            self._traverse(path)
        imageio.imwrite("plots/rrt/robot_traversal.png", self.traverse_map)
        return


def main():
    sim_map = read_pgm('sim_map.pgm')
    sim_map = np.array(sim_map)
    view_points_xy = [np.array([5, 5]), np.array([9, 90]), np.array([90, 90]), np.array([45, 50]), np.array([90, 10])]

    rrt = RRT(view_points_xy, sim_map)
    rrt.find_shortest_paths()
    rrt.traverse()


if __name__ == "__main__":
    main()
