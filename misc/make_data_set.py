import pandas as pd
import random


data = {}
data["km"] = []
data["price"] = []
for i in range (0, 50):
    km = i * 1000 + random.randrange(-40000, 40000)
    data["km"].append(km)
    data["price"].append((km + random.randrange(-20000, 20000)))

df = pd.DataFrame.from_dict(data)
df.to_csv("./data_set.csv", index=False)