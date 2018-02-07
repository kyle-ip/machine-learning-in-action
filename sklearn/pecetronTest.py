# -*- coding: utf-8 -*-
# @Time    : 2018/2/7
# @Author  : yipwinghong
# @Email   : yipwinghong@outlook.com
# @File    : pecetronTest.py      感知机分类测试（广义线性分类模型）
# @Software: PyCharm

from sklearn.datasets import make_classification
from sklearn import cross_validation
from sklearn.linear_model import Perceptron

import matplotlib.pyplot as plt

feature, target = make_classification(      # 随机生成二分类数据集
    n_samples=500,                          # 500个样本
    n_features=2,                           # 2个特征（等于n_informative + n_redundant + n_repeated）
    n_redundant=0,                          # 0个冗余特征
    n_informative=1,                        # 1个多信息特征
    n_clusters_per_class=1,                 # 1个簇/类别
    n_classes=2                             # 2个类别输出
)

training_feature, test_feature, training_target, test_target = cross_validation.train_test_split(
    feature, target, test_size=0.3, random_state=56
)

model = Perceptron()
model.fit(training_feature, training_target)

res = model.predict(test_feature)
print(test_feature)

plt.scatter(test_feature[:, 0], test_feature[:, 1], marker=',')     # 方块样式显示测试数据
for i, txt in enumerate(res):                                       # 预测结果加上标签显示
    plt.annotate(
        txt, (test_feature[:, 0][i], test_feature[:, 1][i])         # 预测值标签，测试数据点坐标
    )
plt.show()