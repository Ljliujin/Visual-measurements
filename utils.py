import numpy as np

def list2corners(corners_list):
    corners = [[corners_list[i]] for i in range(len(corners_list))]
    corners = np.vstack([corners])
    corners=np.array(corners, dtype=np.float32)
    return corners