#!/usr/bin/env python
# coding=utf-8
import operator
import numpy as np
# from pandas import Series
# from collections import namedtuple


def create_data_set():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, data_set, labels, k):
    '''
    k-近邻算法
    inX 为待分类的向量
    data_set 为输入的训练样本集
    labels 为样本集标签
    '''

    # 计算距离(欧氏距离公式 d = pow((x2 - x1)**2 + (y2 - y1)**2, 0.5) )
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(inX, (data_set_size, 1)) - data_set
    distances = np.sum(diff_mat**2, axis=1)**0.5

    sorted_dist_indicies = distances.argsort()

    class_count = {}
    # 选择距离最小的k个点
    for i in range(k):
        vote_i_label = labels[sorted_dist_indicies[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1

    # 排序
    sorted_class_count = sorted(class_count.iteritems(),
                                key=operator.itemgetter(1),
                                reverse=True)
    return sorted_class_count[0][0]
