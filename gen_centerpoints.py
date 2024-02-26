import numpy as np
import pandas as pd



rows = 200
columns = 323
layers = 1

# 输入所在层数，输出本层图片中成正方形的点在点云中所在行的位置的集合
# 输出形式[[0,1,200,201],[1,2,201,202]...]
def oneLayerPointsIdxs(layer):
    b = layer*rows*columns # b = begin
    # 预生成正方形四端点[p1,p2,p3,p4]
    points_indexs = []
    for i in range(rows*(columns-1)):
        points_indexs.append([b+i,b+i+1,b+i+rows,b+i+rows+1])
    # 从中去掉无法形成正方形的点

    points_indexs = np.delete(points_indexs,delete_from_pointsIdx_idx,axis=0)

    return points_indexs

# 输入四点坐标，输出重点值
# 四点位置关系为：
    # (x2,y2)   (x4,y4)
    # (x1,y1)   (x3,y3)
def crossPoint(x1,y1,x2,y2,x3,y3,x4,y4):
    # 四点位置关系为：
    # (x2,y2)   (x4,y4)
    # (x1,y1)   (x3,y3)
    if x1 == x4:
        k1 = None
        b1 = 0
    else:
        k1 = (y4-y1)/(x4-x1)
        b1 = y1-k1*x1

    if x2 == x3:
        k2 = None
        b2 = 0
    else:
        k2 = (y3-y2)/(x3-x2)
        b2 = y2-k2*x2

    if k1 == None:
        x = x1
        y = k2*x + b2
    elif k2 == None:
        x = x2
        y = k1*x+b1
    # elif k1 == k2:
    #     x = np.nan
    #     y = np.nan
    else:
        x = (b2-b1) / (k1-k2)
        y = k1*x + b1
    return [x,y]

# 输入
def CenterPoint(points_indexs):
    center_points = []
    for i in points_indexs:
        centers = []
        # 8个相机的xy，和1个世界坐标点的xyz
        for j in range(8+1):
            [x1,y1,x2,y2,x3,y3,x4,y4] = [data[i[0]][0+2*j],data[i[0]][1+2*j],data[i[1]][0+2*j],data[i[1]][1+2*j],data[i[2]][0+2*j],data[i[2]][1+2*j],data[i[3]][0+2*j],data[i[3]][1+2*j]]
            center = crossPoint(x1,y1,x2,y2,x3,y3,x4,y4)
            print([x1,y1,x2,y2,x3,y3,x4,y4])
            centers.append(center[0])
            centers.append(center[1])
        # 补充上z轴数据
        centers.append(data[i[0]][18])
        np.ravel(centers)
        center_points.append(centers)
    return center_points

def deleteFromPointsIdx():
    delete_from_pointsIdx_idx= []
    for column in range(1,columns):
        delete_from_pointsIdx_idx.append(rows*column-1)
    return delete_from_pointsIdx_idx

print("reading data")
data = np.loadtxt('../pic200x323_3.81_2/data/pointscloud_22.csv', delimiter=',')

delete_from_pointsIdx_idx = deleteFromPointsIdx()
for layer in range(layers):
    print("**********")
    print("layer {}".format(layer))
    points_indexs = oneLayerPointsIdxs(layer)
    centerpoints = CenterPoint(points_indexs)
    print("centerpoints generated,len {}".format(len(centerpoints)))
    dst_df = pd.DataFrame(centerpoints, columns=None)
    dst_df.to_csv("../data/centerpoints.csv", sep=",",header=0,index=False,mode="a")
    print("csv written")

centerpoints = pd.read_csv("../data/centerpoints.csv",header=None)
print("all length {} should be {}".format(centerpoints.shape,(rows-1)*(columns-1)*layers))