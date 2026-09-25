import traffic_sim
import pandas as pd

full_demand = pd.read_excel('SiouxFalls_net_DDMproj.xlsx', sheet_name = 'Full Demand')
network_data = pd.read_excel('SiouxFalls_net_DDMproj.xlsx', sheet_name = 'SiouxFalls_net', skiprows = 7)
link_data = pd.read_excel('SiouxFalls_net_DDMproj.xlsx', sheet_name = 'Link_Length')
link_data['Capacity'] = link_data['number_of_lanes'] * link_data['lane_capacity_in_vhc_per_hour']

test_demand = full_demand[full_demand['DOW'] == 'Friday']
links = [tuple(x) for x in network_data[['Init node', 'Term node']].to_numpy()]
capacity = link_data['Capacity'].tolist()
free_flow_time = network_data['Free flow time'].tolist()

sim = traffic_sim.RoadNetwork(links, 720, 1260, free_flow_time, capacity,test_demand)
sim.run_simulation()


