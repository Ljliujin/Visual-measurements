import numpy as np
import os
import pandas as pd

save_dir = '../pic200x323_3.81_2/data'
data_name = 'interpolate_8.csv'
row = 200
column = 323
layer = 10

def load_data():
    print('loading data')
    imgpts = pd.read_csv('../pic200x323_3.81_2/data/imagepoints_8.csv', header=None)
    wrdpts = pd.read_csv('../pic200x323_3.81_2/data/worldpoints_8.csv', header=None)
    all_csv = np.hstack((imgpts, wrdpts))
    all_csv = pd.DataFrame(np.array(all_csv),columns=["0x","0y","1x","1y","2x","2y","3x","3y","4x","4y","5x","5y","6x","6y","7x","7y","x","y","z"])
    return all_csv


def interpolate_progress(src_df):
    drops = [i*row-1 for i in range(1, column*layer)]
    print(f'drops[0]={drops[0]} drops[-1]={drops[-1]}')
    src_np = np.array(src_df)
    dst_np = np.delete(src_np, drops,axis=0)
    dst_df = pd.DataFrame(dst_np, columns=src_df.columns)
    return dst_df

def gen_data(data):
    print('linear data generating')
    nan = np.where(np.empty_like(data.values), np.nan, np.nan)
    temp = np.hstack([data.values, nan]).reshape(-1, data.shape[1])
    res = pd.DataFrame(temp, columns=data.columns)
    res.drop([len(res)-1], inplace=True)
    res = res.interpolate(method='linear', limit_direction='forward', axis=0)
    target = res[res.index%2==1]
    print('unresonable data droping')
    target = interpolate_progress(target)
    print(f'unresonable data droped\nlinear data shape = {target.shape}')
    save_path = os.path.join(save_dir, data_name)
    target.to_csv(save_path, index=False)
    print(f'linear data saved at {save_path}')

def main():
    data = load_data()
    gen_data(data)

if __name__ == '__main__':
    main()