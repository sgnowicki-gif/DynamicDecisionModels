import pandas as pd
import numpy as np
import networkx as nx

class RoadNetwork():

    #Links is a list of tuples showing connections between nodes

    #Links can have more information about the links in this order:

    #Start and end hour are given in minutes
    #Free_flow_times and capacity much match up with links

    def __init__(self, links, start_hour, end_hour, free_flow_times, capacity):

        #Keeps track of time during the simulation
        self.time = start_hour

        #Current paths keeps track of all paths which are being driven
        self.current_paths = []

        self.big_matrix = np.zeros((len(links), end_hour - start_hour), dtype = int)

        self.link_to_row = {
            link_idx: idx for idx, link_idx in enumerate(links)
        }

        self.row_to_link = {
            v: k for k, v in self.link_to_row.items()
        }

        #These require the actual link not the indexes
        self.link_free_flow_time = dict(zip(links, free_flow_times))
        self.link_capacity = dict(zip(links, capacity))

        #Creates a graph to find shortests routes with
        self.graph = nx.DiGraph()
        for l, length in zip(links, free_flow_times):
            self.graph.add_edge(l[0], l[1], length=length)

    def get_slowdown_factor(self, exceedance):
        #Adjust significnatly later
        return 1 + exceedance / 10

    def update_big_matrix(self, old_path, new_path):
        # Remove old path
        for link, start_time, end_time in old_path:
            row = self.link_to_row[link]
            self.big_matrix[row, int(start_time):int(end_time)] -= 1

        # Add new path
        for link, start_time, end_time in new_path:
            row = self.link_to_row[link]
            self.big_matrix[row, int(start_time):int(end_time)] += 1

    def add_new_driver(self, o, d, start_time):
        #Finds the shortests path for a new driver
        path = nx.shortest_path(
            self.graph,
            source = o,
            target = d,
            weight = 'length'
        )

        #Writes that path a list of links [(n1, n2), (n1,n2)...]
        link_path = list(zip(path[:1], path[1:]))

        #Loops through the list and updates the bigmatrix to show demand at each point in time
        #Also stores a list [(link, start, end)] to store to current paths
        link_path_with_time = []
        cur_time = start_time
        for link in link_path:
            flow_time = self.link_free_flow_time[link]
            link_idx = self.link_to_row[link]

            
            link_path_with_time.append((link, cur_time, cur_time + flow_time))

            self.big_matrix[link_idx, int(cur_time):int(cur_time+flow_time)] += 1

            cur_time += flow_time
        self.current_paths.append(link_path_with_time)

        #Checks for congestion on the big matrix now -- might want to move elsewhere and check every like 10 drivers or every minute
        self.check_big_matrix()

    def check_big_matrix(self):

        #Checks big matrix for places which are exceeding capacity
        capacities = np.array([self.link_capacity[i] for i in range(len(self.big_matrix))])
        exceeds = self.big_matrix > capacities[:, None]

        #If no where is exceeding just stops
        if not exceeds.any():
            return

        #Gets all link indexes and times where exceedance is happening
        link_idxs, times = np.where(capacities)

        #Loops through all these links to find all paths which go through congestion
        for link_idx, t in zip(link_idxs, times):
            target_link = self.row_to_link[link_idx]

            #Finds how large the exceedance is and computes a slowdown factor
            cur_demand = self.big_matrix[link_idx, t]
            capacity = self.link_capacity[target_link]
            extra = cur_demand - capacity
            slowdown_factor = self.get_slowdown_factor()

            #Finds all paths which will be affected by the slowdown
            matching_paths = [i for i , path in enumerate(self.current_paths) if any(link == target_link and start <= t < end for link, start, end in path)]

            #Now updates big matrix and current paths
            for idx in matching_paths:

                path = self.current_paths[idx]

                start_time = path[0][1]

                new_path = []

                cur_time = start_time
                for link, start, end in path:
                    new_time = self.link_free_flow_time[link] * slowdown_factor

                    new_path.append([link, cur_time, cur_time + new_time])

                    cur_time+=new_time

                self.update_big_matrix(path, new_path)
                self.current_paths[idx] = new_path

                









