import pandas as pd
import numpy as np
from scipy.stats import poisson

df = pd.read_excel('SiouxFalls_net_DDMproj.xlsx', sheet_name = 'Full Demand')

df['25th'] = poisson.ppf(0.25, mu = df['Demand']).astye(int)
df['75th'] = poisson.ppf(0.75, mu = df['Demand']).astype(int)

df = df.rename(columns = {'Demand': '50th'})


