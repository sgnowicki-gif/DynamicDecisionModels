import pandas as pd
import numpy as np
import networkx as nx
import time

#This class models the arrival rates for each origin and destination pairs
class ArrivalProcess():
    def __init__(self, vph, seed):

        if vph == 0:
            self.no_demand = True
        else:
            self.rate_per_min = vph / 60
            self.mean_interarrival = 1 / self.rate_per_min
            self.rng = np.random.default_rng(seed)
            self.no_demand = False

    def next_arrival(self):
        if self.no_demand:
            return np.inf
        return self.rng.exponential(self.mean_interarrival)

class RoadNetwork():

    #Links is a list of tuples showing connections between nodes

    #Links can have more information about the links in this order:

    #Start and end hour are given in minutes
    #Free_flow_times and capacity much match up with links

    #Demand is given as a dataframe with columns origin node, destination node, hour, demand

    def __init__(self, links, start_hour, end_hour, free_flow_times, capacity, demand, seed = 1):

        #Keeps track of time during the simulation
        self.time = start_hour
        self.start = start_hour
        self.end = end_hour
        self.seed = seed

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

        #Creates a quick path lookup to speed up computation
        self.path_lookup = self.build_path_lookup()

        #Creates a dictionary which stores an object which can get how long until an agent starts a trip between two nodes
        self.agent_creator = {
            (row['Node 1'], row['Node 2'], row['Hour']): ArrivalProcess(row['Demand'], seed) for _, row in demand.iterrows()
        }

        #Tracks how much time is left between two nodes
        self.time_left = {
            (n1, n2): v.next_arrival() for (n1, n2, h), v in self.agent_creator.items() if h == int(start_hour / 60)
        }

        #Tracks extra time added due to slowdowns
        self.slowdown_time = 0

        #Other data tracked
        self.total_trips = 0
        self.trip_lengths = []

    def __str__(self):
        outputString = ""
        outputString += f"Network with {len(self.big_matrix)} links created"
        outputString += f"Simulation running from {int(self.start / 60)}:{int(self.start % 60)} to {int(self.end / 60)}:{int(self.end % 60)} ({self.end - self.start} minutes)"
        return outputString

    def get_slowdown_factor(self, exceedance):
        #Adjust significnatly later
        return 1 + exceedance / 10

    def update_big_matrix(self, old_path, new_path):
        # Remove old path
        for link, start_time, end_time in old_path:
            row = self.link_to_row[link]
            self.big_matrix[row, int(start_time - self.start):int(end_time - self.start)] -= 1

        # Add new path
        for link, start_time, end_time in new_path:
            row = self.link_to_row[link]
            self.big_matrix[row, int(start_time - self.start):int(end_time - self.start)] += 1

    def build_path_lookup(self):
        path_lookup = {}

        for o in self.graph.nodes:
            for d in self.graph.nodes:
                if o == d:
                    continue

                path = nx.shortest_path(
                    self.graph,
                    source = o,
                    target = d,
                    weight = 'length'
                )

                link_path = list(zip(path[:-1], path[1:]))

                path_lookup[(o,d)] = link_path
        return path_lookup

    def add_new_driver(self, o, d, start_time):
        link_path = self.path_lookup[(o,d)]

        #Loops through the list and updates the bigmatrix to show demand at each point in time
        #Also stores a list [(link, start, end)] to store to current paths
        link_path_with_time = []
        cur_time = start_time
        for link in link_path:
            flow_time = self.link_free_flow_time[link]
            link_idx = self.link_to_row[link]

            
            link_path_with_time.append((link, cur_time, cur_time + flow_time))

            self.big_matrix[link_idx, int(cur_time - self.start):int(cur_time+flow_time - self.start)] += 1

            cur_time += flow_time
        self.current_paths.append(link_path_with_time)

    def check_big_matrix(self):
        #Checks big matrix for places which are exceeding capacity
        capacities = np.array([self.link_capacity[self.row_to_link[i]] for i in range(len(self.big_matrix))])
        exceeds = self.big_matrix > capacities[:, None]

        #If no where is exceeding just stops
        if not exceeds.any():
            return

        #Gets all link indexes and times where exceedance is happening
        link_idxs, times = np.where(exceeds)

        #Loops through all these links to find all paths which go through congestion
        for link_idx, t in zip(link_idxs, times):
            target_link = self.row_to_link[link_idx]

            #Finds how large the exceedance is and computes a slowdown factor
            cur_demand = self.big_matrix[link_idx, t]
            capacity = self.link_capacity[target_link]
            extra = cur_demand - capacity
            slowdown_factor = self.get_slowdown_factor(extra)

            #Finds all paths which will be affected by the slowdown
            matching_paths = [i for i , path in enumerate(self.current_paths) if any(link == target_link and start <= t < end for link, start, end in path)]

            #Now updates big matrix and current paths
            for idx in matching_paths:

                path = self.current_paths[idx]

                start_time = path[0][1]

                new_path = []

                cur_time = start_time
                for link, start, end in path:
                    free_time = self.link_free_flow_time[link]
                    new_time = free_time * slowdown_factor
                    self.slowdown_time += new_time - free_time

                    new_path.append([link, cur_time, cur_time + new_time])

                    cur_time+=new_time

                self.update_big_matrix(path, new_path)
                self.current_paths[idx] = new_path

    def update(self):
        self.time += 1

        if self.time % 5 == 0:
            old_length = len(self.current_paths)
            self.current_paths = [path for path in self.current_paths if path[-1][2] > self.time]
            self.total_trips += old_length - len(self.current_paths)

        self.time_left = {k: v - 1 for k, v in self.time_left.items()}
        update_od_pairs = [k for k, v in self.time_left.items() if v < 1]
        for n1, n2 in update_od_pairs:
            self.add_new_driver(n1, n2, self.time)
            cur_hour = int(self.time / 60)
            new_time = self.agent_creator[(n1, n2, cur_hour)].next_arrival()
            inter_minute_time = 0
            while new_time < 1 and inter_minute_time < 1:
                self.add_new_driver(n1, n2, self.time)
                inter_minute_time += new_time
                new_time = self.agent_creator[(n1, n2, cur_hour)].next_arrival()
            self.time_left[(n1, n2)] = new_time
    
        self.check_big_matrix()

        if self.time % 60 == 0:
            print(f"Took time {time.time() - self.last_update_time}")
            print(f"At time {int(self.time / 60)}:00")
            print(f"Completed {self.time - self.start:.2f} minutes ({(self.time - self.start) / (self.end - self.start):.2f}% completed)")
            print(f"Completed {self.total_trips} with {self.slowdown_time} total slowdown minutes")
            print(f"Total Demand = {np.sum(self.big_matrix)}")

            self.last_update_time = time.time()

    def run_simulation(self):
        self.start_sim_time = time.time()
        self.last_update_time = self.start_sim_time
        while self.time < self.end:
            self.update()
        print(f"Simulation Complete in {time.time() - self.start_sim_time:.2f}")
        print(f"Total slowdown time: {self.slowdown_time}")

    def reset(self):
        self.slowdown_time = 0
        self.time = self.start
        self.time_left = {(n1, n2): v.next_arrival() for (n1, n2, h), v in self.agent_creator if h == int(self.start / 60)}









