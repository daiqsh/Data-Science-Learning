import pandas as pd
import numpy as np

data = pd.Series(np.random.random_integers(low=0, high=1000, size=1000))
# print(data)
print(data.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))