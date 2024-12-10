import pandas as pd
import random

random.seed()

data = {}
data["km"] = []
data["price"] = []
for i in range (0, 50):
    data["km"].append (random.randrange(0, 100000))
    data["price"].append (random.randrange(0, 100000))
# print(data)

df = pd.DataFrame.from_dict(data)
df.to_csv("rand_data.csv", index=False)