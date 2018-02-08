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


def fill_null(data, features, judge_feature, judge_value, replace_value):
    """
    填补缺失值
    :param data:
    :param features:        特征列表
    :param judge_feature:   判断特征
    :param judge_value:     判断值
    :param replace_value:   替换值
    :return:
    """
    for feature in features:
        null_index = data[data[feature].isnull()].index         # 查找该特征缺失的样本的索引
        sp_index = [i for i in null_index if data[judge_feature][i] != judge_value]     # 查找判断特征对应值不合理的样本的索引
        data[feature].fillna(replace_value, inplace=True)       # 使用替换值填补缺失值
        for i in sp_index:
            data[feature].iloc[i] = data[feature].mode()[0]     # 众数填补缺失值
    return data


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
# null_count(data)

# 缺失值处理
zero_replace_features = [
    'BsmtHalfBath', 'BsmtHalfBath', 'BsmtFullBath',
    'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFinSF2', 'BsmtFinSF1'
]                                                   # 使用0填补
for feature in zero_replace_features:
    data[feature].fillna(0.0, inplace=True)

features = [
    'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2'
]                                                   # 使用'U'填补
data = fill_null(data, features, 'TotalBsmtSF', 0.0, 'U')
data = fill_null(data, ['PoolQC'], 'PoolArea', 0, 'U')


mode_inplace = [
    'MSZoning', 'Utilities', 'Exterior1st', 'Electrical',
    'Exterior2nd', 'KitchenQual', 'SaleType'
]                                                   # 使用众数填补
for feature in mode_inplace:
    data[feature].fillna(data[feature].mode()[0], inplace=True)

g = sns.factorplot(
    x='KitchenAbvGr', y='KitchenQual',
    data=data, kind='box'
)

data['Functional'].fillna('Typ', inplace=True)      # 使用'TYP'填补
plt.show()