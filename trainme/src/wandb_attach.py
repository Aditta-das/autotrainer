import numpy as np
import pandas as pd

df = pd.DataFrame(columns=["label"])
df["label"] = np.arange(1, 100)
print(df)