import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

rng = random.Random(100)

project_start = datetime(2026, 6, 1, 0,0)
project_end = datetime(2026, 9, 30, 0, 0)

full_demand = pd.read_excel('SiouxFalls_net_DDMproj.xlsx', sheet_name = 'Full Demand')

#Start times are exaclty at hour intervals
def generate_agents(start_time, end_time):
    t = start_time
    rows = []
    count = 0
    pair_prev_time = {}
    while t < end_time:
        temp_demand = full_demand[full_demand['DOW'] == t.strftime('%A')]
        temp_demand = temp_demand[temp_demand['Hour'] == t.hour]
        node_pairs = list(zip(temp_demand['Node 1'], temp_demand['Node 2'], temp_demand['Demand']))
        for pair in node_pairs:
            if pair[2] == 0:
                continue
            temp_time = pair_prev_time.get((pair[0], pair[1]), t)
            while temp_time < t + timedelta(hours = 1):
                inter_time = rng.expovariate(pair[2] / 60)
                temp_time += timedelta(minutes = inter_time)
                rows.append({
                    'agent_id': count,
                    'origin_node_id': pair[0],
                    'destination_node_id': pair[1],
                    'departure_time_in_min': temp_time
                })
                count += 1
            pair_prev_time[(pair[0], pair[1])] = temp_time
        t += timedelta(hours = 1)
    outputDf = pd.DataFrame(rows)
    outputDf.to_csv('input_agent.csv')
    print('Completed !!')

generate_agents(project_start, project_end)