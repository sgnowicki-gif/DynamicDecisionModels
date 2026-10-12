import pandas as pd
import numpy as np

demand = pd.read_csv('demand.csv')

promo = "No Promo"

counts = (
    demand.loc[demand["Promotion"] == promo, "Sales Menu 2"]
    .value_counts()
    .sort_index()
)

print(counts)

