import os
import numpy as np
import pandas as pd

interpolate = pd.read_csv("./data/interpolate.csv")
print(interpolate.shape)
column = 200
row = 299
layers = 76
before = len(interpolate)
# 待丢弃的行（由大到小，由下往上丢弃，避免行数变化）
drops = [i for i in range(len(interpolate)-1,0,-1) if (i+1)%column==0]

for drop in drops:
    # print(interpolate.iloc[drop-1:drop+2,:])
    print("droping index={}".format(drop))
    interpolate.drop(drop,inplace=True)

after = len(interpolate)

print(before-after==row*layers-1)

interpolate.to_csv("./data/interpolate_new.csv", sep=",", index=False)