import pandas as pd
import numpy as np

od_df_chart = pd.read_excel('SiouxFalls_net_DDMproj.xlsx', sheet_name = 'Demand')
profile_df = pd.read_excel('SiouxFalls_net_DDMproj.xlsx', sheet_name = 'Demand Time Matrix')

od_df = od_df_chart.melt(id_vars = 'Node 1', var_name = 'Node 2', value_name = 'Demand')

peak_profile = profile_df.iloc[:,1:].to_numpy().max()

profile_long = profile_df.melt(id_vars = 'DOW', var_name = 'Hour', value_name = 'Profile Demand')

profile_long['Factor'] = profile_long['Profile Demand'] / peak_profile

result = od_df.merge(profile_long[['DOW', 'Hour', 'Factor']], how = 'cross')

result['Demand'] = result['Demand'] * result['Factor']

result = result[['Node 1', 'Node 2', 'DOW', 'Hour', "Demand"]]

with pd.ExcelWriter(
    'SiouxFalls_net_DDMproj.xlsx', engine = 'openpyxl', mode = 'a', if_sheet_exists = 'replace'
) as writer:
    result.to_excel(writer, sheet_name = 'Full Demand', index = False)