# -*- coding: utf-8 -*-
# @Time    : 2018/2/7
# @Author  : yipwinghong
# @Email   : yipwinghong@outlook.com
# @File    : boston.py      支持向量机分类测试（波士顿房产）
# @Software: PyCharm

import os

import pandas as pd
import numpy as np
import seaborn as sns                   # 更友好的绘图库
import warnings

from sklearn import datasets
from sklearn import cross_validation
from sklearn.svm import LinearSVR

from matplotlib import pyplot as plt

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# boston = datasets.load_boston()
# # print(boston.DESCR)
#
# feature, target = boston.data, boston.target
#
# model = LinearSVR()
# predictions = cross_validation.cross_val_predict(           # 交叉验证
#     model, feature, target, cv=10                           # 随机将数据集等分成10份（9份用作训练，1份用作测试，循环验证）
# )
#
# plt.scatter(target, predictions)
# plt.plot(                                                   # 绘制 45 度参考线
#     [target.min(), target.max()], [target.min(), target.max()],
#     "k--", lw=4
# )
# plt.xlabel("true_target")
# plt.ylabel("prediction")
#
# plt.show()


def null_count(data):
    """
    统计缺失值：删除数目为0的特征，降序排列
    :param data:
    :return:        所有特征的缺失值个数
    """
    null_data = data.isnull().sum()     # 计算缺失值数量
    null_data = null_data.drop(null_data[null_data == 0].index).sort_values(ascending=False)

    return null_data


# 忽略Warnings
def ignore(*args, **kwargs):
    pass

warnings.warn = ignore


# 加载数据
training_set = pd.read_csv(os.path.join(path, "data/boston_train.csv"))
test_set = pd.read_csv(os.path.join(path, "data/boston_test.csv"))
training_set.drop(['Id'], axis=1, inplace=True)            # 移除Id列，axis=0表示行，axis=1表示列，inplace表示在原DF上修改
test_set.drop(['Id'], axis=1, inplace=True)

# 离群点：删除居住面积大于4000的数据
training_set.drop(training_set[training_set['GrLivArea'] > 4000].index, inplace=True)

sns.set(style='darkgrid')

# 打印散点图
# fig = plt.figure()
# ax = plt.scatter(training_set['GrLivArea'], training_set['SalePrice'])
#
# plt.xlabel('GrLivArea')
# plt.ylabel('SalePrice')

# plt.show()

# 打印房价分布曲线
# training_set['SalePrice'] = np.log(training_set['SalePrice'])       # Log Transformation（处理右偏态分布）
# g = sns.distplot(                                                   # 绘制柱状图
#     training_set['SalePrice'],
#     kde=True,                                                       # 绘制拟合曲线
#     label='skewness:%.2f' % training_set['SalePrice'].skew()        # skew()函数计算偏态系数
# )
# plt.legend(loc='best', fontsize='large')
# g.set(xlabel='SalePrice')

# plt.show()

# 计算关联系数
# plt.subplots(figsize=(20, 16))
# sns.heatmap(training_set.corr(), square=True)
# plt.show()

# 合并训练集和测试集
data = pd.concat([training_set, test_set], axis=0, ignore_index=True)
null_count(data)