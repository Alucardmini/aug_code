# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/9/21 11:22 AM'


import pandas as pd


def load_data():
    train_data = {}
    file_path = 'tiny_train_input.csv'

    data = pd.read_csv(file_path, header=None)
    data.columns = ['c' + str(i) for i in range(data.shape[1])]

    label = data.c0.values
    label = label.reshape(len(label), 1)

    train_data['y_train'] = label

    co_feature = pd.DataFrame()  # 归一化的特征
    ca_feature = pd.DataFrame()

    ca_col = []
    co_col = []

    feat_dict = {}
    cnt = 1

    for i in range(1, data.shape[1]):
        col_data = data.iloc[:, i]  # 取第i列数据
        col_name = col_data.name
        value_length = len(set(col_data))

        if value_length > 10:
            col_data = (col_data - col_data.mean()) / col_data.std()
            co_feature = pd.concat([co_feature, col_data], axis=1)

            feat_dict[col_name] = cnt
            cnt += 1
            co_col.append(col_name)

        else:
            us = col_data.unique()
            feat_dict[col_name] = dict(zip(us, range(cnt, len(us) + cnt)))
            ca_feature = pd.concat([ca_feature, col_data], axis=1)
            cnt += len(us)
            ca_col.append(col_data)

    feat_dim = cnt

    feature_value = pd.concat([co_feature, ca_feature], axis=1)
    feature_index = feature_value.copy()

    for i in feature_index.columns:
        if i in co_col:
            feature_index[i] = feat_dict[i]
        else:
            feature_index[i] = feature_index[i].map(feat_dict[i])
            feature_value[i] = 1
    train_data['xi'] = feature_index.values.tolist()
    train_data['xv'] = feature_value.values.tolist()
    train_data['feat_dim'] = feat_dim
    return train_data


if __name__ == '__main__':
    pass






