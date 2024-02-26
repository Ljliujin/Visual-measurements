import numpy as np
row, column, layer = 200, 323, 151
wrd_points = []
for z in range(layer):
    for x in range(column):
        for y in range(row):
            wrd_points.append([x*3.81,y*3.81,z*2])
            # wrd_points.append([x,y,z])

txt = np.vstack(wrd_points)
np.savetxt('./worldpoints.csv', txt, fmt='%f', delimiter=',')