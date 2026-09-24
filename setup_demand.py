import pandas as pd
import numpy as np

peak_demand_chart = pd.read_excel('SiouxFalls_net_DDMproj.xlsx', sheet_name = 'Demand')
demand_profile = pd.read_excel('SiouxFalls_net_DDMproj.xlsx', sheet_name = 'Demand Time Matrix')

print(peak_demand_chart.head())
print(demand_profile.head())