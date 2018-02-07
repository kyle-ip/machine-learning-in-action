# -*- coding: utf-8 -*-
# @Time    : 2018/2/7
# @Author  : yipwinghong
# @Email   : yipwinghong@outlook.com
# @File    : boston.py      支持向量机分类测试（波士顿房产）
# @Software: PyCharm

from sklearn import datasets
from sklearn import cross_validation
from sklearn.svm import LinearSVR

from matplotlib import pyplot as plt

boston = datasets.load_boston()
# print(boston.DESCR)

feature, target = boston.data, boston.target

model = LinearSVR()
predictions = cross_validation.cross_val_predict(           # 交叉验证
    model, feature, target, cv=10                           # 随机将数据集等分成10份（9份用作训练，1份用作测试，循环验证）
)

plt.scatter(target, predictions)
plt.plot(                                                   # 绘制 45 度参考线
    [target.min(), target.max()],
    [target.min(), target.max()],
    "k--", lw=4
)
plt.xlabel("true_target")
plt.ylabel("prediction")

plt.show()