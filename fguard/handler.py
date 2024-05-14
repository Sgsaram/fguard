import sys

import numpy as np
import tqdm

sys.setrecursionlimit(2048 * 2048)


class Event:
    def __init__(
        self,
        event_type: str,
        time: int,
        component: int,
        prev_connections: list[int] = list(),
    ) -> None:
        self.event_type = event_type
        self.time = int(time)
        self.component = int(component)
        self.prev_connections = list(map(int, prev_connections))

    def get_dict(
        self,
    ) -> dict:
        res = dict()
        res["event_type"] = self.event_type
        res["time"] = self.time
        res["id"] = self.component
        res["previous_connections"] = self.prev_connections
        return res
    

class ComponentFounder:
    def __init__(
        self,
        image: np.ndarray,
        min_comp_dist: int = 1,
    ) -> None:
        self.image = image
        self.comp = np.zeros_like(self.image, dtype=np.uint32)
        self.min_comp_dist = min_comp_dist
        self.__n_size = image.shape[0]
        self.__m_size = image.shape[1]

    def get_neighbours(
        self,
        x: int,
        y: int,
    ) -> list[tuple[int, int]]:
        vertices: list[tuple[int, int]] = list()
        for xp in range(max(0, x - self.min_comp_dist), min(self.__n_size, x + self.min_comp_dist + 1)):
            for yp in range(max(0, y - self.min_comp_dist), min(self.__m_size, y + self.min_comp_dist + 1)):
                if self.image[xp][yp] and not self.comp[xp][yp]:
                    vertices.append((xp, yp))
        return vertices

    def dfs(
        self,
        x: int,
        y: int,
        cur_color: int,
    ) -> None:
        self.comp[x][y] = cur_color
        neighbours = self.get_neighbours(x, y)
        for xp, yp in neighbours:
            self.dfs(xp, yp, cur_color)


    def __call__(
        self,
    ) -> tuple[np.ndarray, int]:
        comp_cnt = 0
        for x in range(self.__n_size):
            for y in range(self.__m_size):
                if self.image[x][y] and not self.comp[x][y]:
                    comp_cnt += 1
                    self.dfs(x, y, comp_cnt)
        return (self.comp, comp_cnt)


class DeforestationHandler:
    def __init__(
        self,
        images: np.ndarray,
        min_comp_dist: int = 1,
    ) -> None:
        self.images = images
        self.images_cnt = images.shape[0]
        self.min_comp_dist = min_comp_dist

    def __call__(
        self,
    ) -> tuple[np.ndarray, int, list[Event]]:
        """
        Returns component cluster and number of unique components
        """
        events = list()
        if self.images_cnt == 0:
            return (np.array([]), 0, events)
        n = self.images.shape[1]
        m = self.images.shape[2]
        res_cluster = []
        prev_cluster = np.zeros_like(self.images[0], dtype=np.uint32)
        unique_comp_cnt = 0
        for cur_time in tqdm.tqdm(range(self.images_cnt), desc="Processing images", unit="image"):
            cmp_handler = ComponentFounder(self.images[cur_time], self.min_comp_dist)
            cur_cluster, comp_cnt = cmp_handler()
            conn_with_prev = [set() for _ in range(comp_cnt + 1)]
            for x in range(n):
                for y in range(m):
                    if cur_cluster[x][y] != 0:
                        assert cur_cluster[x][y] <= comp_cnt, f"{cur_cluster[x][y]} -> {comp_cnt}"
                        conn_with_prev[cur_cluster[x][y]].add(prev_cluster[x][y])
            change_to = [0] * (comp_cnt + 1)
            for i in range(1, comp_cnt + 1):
                connection_list = sorted(list(conn_with_prev[i]))
                if len(connection_list) == 1 and connection_list[0] == 0:
                    unique_comp_cnt += 1
                    change_to[i] = unique_comp_cnt
                    events.append(Event("new", cur_time, change_to[i]))
                elif len(connection_list) == 1 and connection_list[0] != 0:
                    change_to[i] = connection_list[0]
                elif len(connection_list) == 2 and connection_list[0] == 0:
                    change_to[i] = connection_list[1]
                    events.append(Event("add", cur_time, change_to[i]))
                elif len(connection_list) > 2:
                    was_changed = False
                    if connection_list[0] == 0:
                        was_changed = True
                        connection_list.pop(0)
                    change_to[i] = connection_list[0]
                    events.append(Event("merge", cur_time, change_to[i], connection_list))
                    if was_changed:
                        events.append(Event("add", cur_time, change_to[i]))
            for x in range(n):
                for y in range(m):
                    cur_cluster[x][y] = change_to[cur_cluster[x][y]]
            res_cluster.append(cur_cluster)
            prev_cluster = cur_cluster.copy()
        return (np.array(res_cluster), unique_comp_cnt, events)

