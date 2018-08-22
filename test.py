import random

#rom pomegranate import *

import pandas as pd
import numpy as np

pd.set_option('display.expand_frame_repr', False)

df_read = pd.read_csv("DC.Inputs.csv")
df_out = pd.read_csv("DC.Targets.csv")

df_cleaned = df_read

X = df_cleaned.drop(['custAttr2'], axis=1)
print (df_cleaned.dtypes)
print(X)
#print(df_out.head(5))
#df_cleaned = df

#y = df_cleaned.isFraud

random.seed(0)
