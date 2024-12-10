import pandas as pd
import random

data = {}
data["km"] = []
data["price"] = []
for i in range (0, 50):
    data["km"].append(i * 1000)
    data["price"].append(i * 1000 + 500)

df = pd.DataFrame.from_dict(data)
df.to_csv("perfect_data.csv", index=False)