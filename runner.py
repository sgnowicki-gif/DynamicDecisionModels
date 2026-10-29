#How to use
# import sys
# sys.path.insert(0, "/path/to/parent/of/pysim_construction")
# from pysim_construction import run_pysim_construction
# result = run_pysim_construction("my_inputs", "my_outputs", emissions=False)

import sys
from pysim_construction import run_pysim_construction
import pandas as pd

full_demand = pd.read_excel('SiouxFalls_net_DDMproj.xlsx', sheet_name = 'Full Demand')
network_data = pd.read_excel('SiouxFalls_net_DDMproj.xlsx', sheet_name = 'SiouxFalls_net', skiprows = 7)
link_data = pd.read_excel('SiouxFalls_net_DDMproj.xlsx', sheet_name = 'Link_Length')
link_data['Capacity'] = link_data['number_of_lanes'] * link_data['lane_capacity_in_vhc_per_hour']




