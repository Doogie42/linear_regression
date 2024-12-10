import numpy as np
import pandas as pd
import sys

def load(path: str) -> pd.DataFrame:
    """Read csv file and return it as a panda Dataframe.
    Rais ValueError if problem"""
    if path.rfind(".csv") == -1:
        raise ValueError("Wrong type of file")
    try:
        data = pd.read_csv(path)
    except Exception:
        raise ValueError("Wrong encoding of file")
    print(f"Loading dataset of dimension {data.shape}")
    return data

if sys.argv == 1:
    data = load("rand_data.csv")
else:
    data = load(sys.argv[1])
(m, b),(residual,), *_ = np.polyfit(data["km"], data["price"], deg=1, full=True)
print("theta0 ", m)
print("theta1 ", b)